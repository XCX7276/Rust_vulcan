//! Отрисовка ландшафта (чанков).
//!
//! Устройство:
//! - **пул геометрии** — один большой буфер видеопамяти, где лежат квады всех чанков подряд;
//!   место выдаёт `RangeAllocator` (единица — один квад = 8 байт);
//! - **таблица чанков** — буфер, где для каждого «слота» чанка лежит запись 32 байта: угол чанка
//!   (относительно точки отсчёта рендера), признак «живой» вместе с масштабом (узлы LOD: клетка
//!   2^L блоков), первый квад в пуле и число квадов каждой из 6 сторон. Её читают шейдер отсечения (`cull.comp`) и вершинный шейдер
//!   (по `gl_InstanceIndex`);
//! - **цвета блоков** — TEMPORARY: плоский цвет на блок, пока нет текстур;
//! - **конвейер (pipeline)** — шейдеры `terrain.vert/frag`, глубина, отсечение задних граней.
//!
//! Вершин в обычном смысле нет (vertex pulling): шейдер сам читает квад и строит 6 вершин.
//! Что рисовать, решает видеокарта (`culling.rs`): она пишет команды
//! `(квады × 6, 1, первый_квад × 6, слот)`, и ландшафт рисуется одним вызовом
//! `vkCmdDrawIndirectCount` на проход.
//!
//! **Загрузка мешей без ожидания.** `queue_upload` сразу выдаёт место в пуле и слот, а данные
//! кладёт в очередь. В начале каждого кадра (`record_uploads`, после того как видеокарта закончила
//! прошлый кадр этого слота) данные копируются в «свою» часть промежуточного буфера и в команды
//! кадра записываются копирования — не больше бюджета байт за кадр. Запись в таблице чанков
//! пишется в том же кадре, и с него чанк участвует в отсечении. Процессор никогда не ждёт видеокарту.
//!
//! **Удаление.** Запись чанка в таблице обнуляется в начале ближайшего кадра (после этого кадра
//! видеокарта его не рисует), а место в пуле и слот освобождаются, когда этот кадр завершён.

use std::collections::VecDeque;
use std::io::Cursor;

use anyhow::{Context as _, Result, bail};
use ash::{Device, vk};
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::culling::{DRAWS_PER_CHUNK, IndirectList};
use crate::range_alloc::RangeAllocator;
use crate::resources::{GpuBuffer, SharedAllocator};

/// Вместимость пула геометрии в квадах (8 байт на квад → 64 МБ).
pub const POOL_QUADS: u32 = 8 * 1024 * 1024;
/// Наибольший уровень LOD (клетка 2^уровень блоков), который понимают шейдеры.
pub const MAX_LOD_LEVEL: u32 = 4;
/// Сколько чанков с мешем может быть одновременно.
pub const MAX_CHUNK_SLOTS: u32 = 65_536;
/// Часть промежуточного буфера на один кадр «в полёте» = бюджет загрузки за кадр.
pub const UPLOAD_BUDGET_BYTES: u64 = 8 * 1024 * 1024;
const QUAD_BYTES: u64 = 8;
/// Запись чанка в таблице (`ChunkRecord` в шейдерах).
const SLOT_BYTES: u64 = 32;

const TERRAIN_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/terrain.vert.spv"));
const TERRAIN_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/terrain.frag.spv"));

/// Данные, которые каждый кадр передаются шейдерам ландшафта (ровно 128 байт — максимум,
/// который Vulkan гарантирует для push constants).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TerrainPush {
    pub view_proj: [f32; 16],
    pub camera_block: [i32; 4],
    pub camera_frac: [f32; 4],
    pub sun_dir: [f32; 4],
    pub fog_color: [f32; 4],
}

/// Номер чанка в рендере (= его слот в таблице чанков).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkMeshId(pub u32);

/// Итог постановки меша в очередь загрузки.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkUpload {
    /// Место выдано; чанк появится на экране, когда данные скопируются (обычно в ближайший кадр).
    Queued(ChunkMeshId),
    /// Меш пустой — рисовать нечего.
    Empty,
    /// Пул геометрии или таблица чанков переполнены.
    PoolFull,
}

#[derive(Clone, Copy)]
struct ChunkDraw {
    first_quad: u32,
    quad_count: u32,
    /// Данные уже скопированы в видеопамять — можно рисовать.
    ready: bool,
}

/// Меш, ждущий копирования в видеопамять.
struct PendingUpload {
    slot: u32,
    /// Запись для таблицы чанков.
    record: [u32; 8],
    quads: Vec<u32>,
}

/// Счётчики для экрана F3.
#[derive(Clone, Copy, Debug, Default)]
pub struct TerrainStats {
    pub chunks: u32,
    pub quads_resident: u64,
    pub pool_used_quads: u32,
    pub pool_capacity_quads: u32,
    /// Мешей в очереди на копирование.
    pub pending_uploads: u32,
}

pub struct Terrain {
    pool: GpuBuffer,
    chunk_table: GpuBuffer,
    colors: GpuBuffer,
    /// Промежуточный буфер: по `UPLOAD_BUDGET_BYTES` на каждый кадр «в полёте».
    staging: GpuBuffer,
    frames_in_flight: usize,
    ranges: RangeAllocator,
    free_slots: Vec<u32>,
    draws: Vec<Option<ChunkDraw>>,
    pending: VecDeque<PendingUpload>,
    live_chunks: u32,
    resident_quads: u64,
    /// Удалённые чанки, чью запись в таблице ещё надо обнулить в ближайшем кадре.
    pending_clears: Vec<(u32, ChunkDraw)>,
    /// Чанки, чья запись обнулена в кадре N: (кадр, слот, меш). Слот и место в пуле освобождаются,
    /// только когда видеокарта точно закончила этот кадр (она ещё может читать эти данные).
    retired: Vec<(u64, u32, ChunkDraw)>,
    /// Номер наибольшего когда-либо занятого слота + 1: столько записей проверяет отсечение.
    slot_limit: u32,
    /// Таблица чанков ещё не обнулена (содержимое новой видеопамяти не определено).
    needs_clear: bool,
    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    face_culling: bool,
}

impl Terrain {
    pub fn new(
        device: &Device,
        allocator: &SharedAllocator,
        color_format: vk::Format,
        depth_format: vk::Format,
        face_culling: bool,
        frames_in_flight: usize,
    ) -> Result<Self> {
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let pool = GpuBuffer::new(
            device,
            allocator,
            POOL_QUADS as u64 * QUAD_BYTES,
            storage,
            MemoryLocation::GpuOnly,
            "terrain quad pool",
        )?;
        let chunk_table = GpuBuffer::new(
            device,
            allocator,
            MAX_CHUNK_SLOTS as u64 * SLOT_BYTES,
            storage,
            MemoryLocation::GpuOnly,
            "terrain chunk table",
        )?;
        let colors = GpuBuffer::new(
            device,
            allocator,
            u16::MAX as u64 * 16,
            storage,
            MemoryLocation::GpuOnly,
            "block colors",
        )?;
        let staging = GpuBuffer::new(
            device,
            allocator,
            UPLOAD_BUDGET_BYTES * frames_in_flight as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "terrain staging",
        )?;

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
            })
            .collect();
        // SAFETY: valid device; objects are destroyed in `destroy`.
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }?;
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3,
        }];
        // SAFETY: valid device.
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }?;
        let layouts = [set_layout];
        // SAFETY: pool has room for exactly this set.
        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
        }?[0];
        let infos = [
            [vk::DescriptorBufferInfo::default().buffer(pool.buffer).range(vk::WHOLE_SIZE)],
            [vk::DescriptorBufferInfo::default().buffer(chunk_table.buffer).range(vk::WHOLE_SIZE)],
            [vk::DescriptorBufferInfo::default().buffer(colors.buffer).range(vk::WHOLE_SIZE)],
        ];
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        // SAFETY: the set and buffers are valid; the set is not in use yet.
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let push_range = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: size_of::<TerrainPush>() as u32,
        }];
        // SAFETY: valid device and set layout.
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&push_range),
                None,
            )
        }?;
        let pipeline =
            create_pipeline(device, pipeline_layout, color_format, depth_format, face_culling)?;

        Ok(Self {
            pool,
            chunk_table,
            colors,
            staging,
            frames_in_flight,
            ranges: RangeAllocator::new(POOL_QUADS),
            free_slots: (0..MAX_CHUNK_SLOTS).rev().collect(),
            draws: vec![None; MAX_CHUNK_SLOTS as usize],
            pending: VecDeque::new(),
            live_chunks: 0,
            resident_quads: 0,
            pending_clears: Vec::new(),
            retired: Vec::new(),
            slot_limit: 0,
            needs_clear: true,
            set_layout,
            descriptor_pool,
            descriptor_set,
            pipeline_layout,
            pipeline,
            face_culling,
        })
    }

    /// Пересоздаёт конвейер под новый формат картинок (бывает при смене монитора/HDR).
    pub fn recreate_pipeline(
        &mut self,
        device: &Device,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<()> {
        let pipeline = create_pipeline(
            device,
            self.pipeline_layout,
            color_format,
            depth_format,
            self.face_culling,
        )?;
        // SAFETY: called after device_wait_idle when the swapchain is rebuilt.
        unsafe { device.destroy_pipeline(self.pipeline, None) };
        self.pipeline = pipeline;
        Ok(())
    }

    /// Загружает цвета блоков (линейные RGBA). Вызывается один раз при запуске, до первого кадра,
    /// поэтому здесь допустимо дождаться видеокарты. TEMPORARY: до появления текстур.
    pub fn set_block_colors(
        &mut self,
        device: &Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        colors: &[[f32; 4]],
    ) -> Result<()> {
        let bytes: &[u8] = bytemuck::cast_slice(colors);
        if bytes.len() as u64 > self.colors.size || bytes.len() as u64 > UPLOAD_BUDGET_BYTES {
            bail!("too many block colors");
        }
        self.write_staging(0, bytes)?;
        let region = [vk::BufferCopy::default().size(bytes.len() as u64)];
        // SAFETY: init-time only; we wait for the queue before returning.
        unsafe {
            let cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_copy_buffer(cmd, self.staging.buffer, self.colors.buffer, &region);
            record_upload_barrier(device, cmd);
            device.end_command_buffer(cmd)?;
            let buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let submit = vk::SubmitInfo2::default().command_buffer_infos(&buffers);
            device.queue_submit2(queue, &[submit], vk::Fence::null())?;
            device.queue_wait_idle(queue)?;
            device.free_command_buffers(command_pool, &[cmd]);
        }
        Ok(())
    }

    /// Ставит меш чанка в очередь загрузки. Возвращает сразу, видеокарту не ждёт.
    /// `faces` — число квадов каждой стороны; квады в `quads` идут группами в том же порядке.
    /// `lod_level` — 0 для обычного чанка, L для узла LOD (клетка = 2^L блоков).
    pub fn queue_upload(
        &mut self,
        origin: [i32; 3],
        quads: Vec<u32>,
        faces: [u32; 6],
        lod_level: u32,
    ) -> ChunkUpload {
        let quad_count = (quads.len() / 2) as u32;
        if quad_count == 0 {
            return ChunkUpload::Empty;
        }
        if faces.iter().sum::<u32>() != quad_count || faces.iter().any(|&n| n > u16::MAX as u32) {
            log::error!("chunk mesh face counts {faces:?} do not match {quad_count} quads");
            return ChunkUpload::Empty;
        }
        if quad_count as u64 * QUAD_BYTES > UPLOAD_BUDGET_BYTES {
            log::warn!("chunk mesh with {quad_count} quads is larger than the upload budget");
            return ChunkUpload::PoolFull;
        }
        let Some(first_quad) = self.ranges.alloc(quad_count) else {
            return ChunkUpload::PoolFull;
        };
        let Some(slot) = self.free_slots.pop() else {
            self.ranges.free(first_quad, quad_count);
            return ChunkUpload::PoolFull;
        };
        self.draws[slot as usize] = Some(ChunkDraw {
            first_quad,
            quad_count,
            ready: false,
        });
        self.live_chunks += 1;
        self.resident_quads += quad_count as u64;
        self.slot_limit = self.slot_limit.max(slot + 1);
        let record = [
            origin[0] as u32,
            origin[1] as u32,
            origin[2] as u32,
            1 + lod_level.min(MAX_LOD_LEVEL), // живой; масштаб клетки = 2^(значение − 1)
            first_quad,
            faces[0] | faces[1] << 16,
            faces[2] | faces[3] << 16,
            faces[4] | faces[5] << 16,
        ];
        self.pending.push_back(PendingUpload {
            slot,
            record,
            quads,
        });
        ChunkUpload::Queued(ChunkMeshId(slot))
    }

    /// Убирает чанк. Если его данные ещё не были скопированы — место освобождается сразу;
    /// иначе запись в таблице обнулится в ближайшем кадре, а место — когда тот кадр завершится.
    pub fn remove(&mut self, id: ChunkMeshId) {
        let Some(draw) = self.draws[id.0 as usize].take() else {
            return;
        };
        self.live_chunks -= 1;
        self.resident_quads -= draw.quad_count as u64;
        if draw.ready {
            self.pending_clears.push((id.0, draw));
        } else {
            self.pending.retain(|p| p.slot != id.0);
            self.ranges.free(draw.first_quad, draw.quad_count);
            self.free_slots.push(id.0);
        }
    }

    /// Вызывается, когда видеокарта завершила все кадры до `completed_frame` включительно.
    pub fn release_retired(&mut self, completed_frame: u64) {
        let free_slots = &mut self.free_slots;
        let ranges = &mut self.ranges;
        self.retired.retain(|&(frame, slot, draw)| {
            if frame <= completed_frame {
                ranges.free(draw.first_quad, draw.quad_count);
                free_slots.push(slot);
                false
            } else {
                true
            }
        });
    }

    /// Записывает в `cmd` изменения таблицы чанков (удаления и новые записи) и копирование
    /// ожидающих мешей из промежуточного буфера кадра `frame_slot` (не больше бюджета).
    /// `frame_number` — номер записываемого кадра. Вызывать после ожидания fence этого слота,
    /// до отсечения и отрисовки. Возвращает число скопированных байт мешей.
    pub fn record_uploads(
        &mut self,
        device: &Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        frame_number: u64,
    ) -> Result<u64> {
        let mut table_written = false;
        if self.needs_clear {
            let after_fill = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
            // SAFETY: cmd is recording outside a rendering scope.
            unsafe {
                device.cmd_fill_buffer(cmd, self.chunk_table.buffer, 0, vk::WHOLE_SIZE, 0);
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&[after_fill]),
                );
            }
            self.needs_clear = false;
            table_written = true;
        }
        // Удалённые чанки: обнуляем запись (с этого кадра отсечение их пропускает).
        for (slot, draw) in self.pending_clears.drain(..) {
            // SAFETY: cmd is recording outside a rendering scope; the region lies within the table.
            unsafe {
                device.cmd_update_buffer(
                    cmd,
                    self.chunk_table.buffer,
                    slot as u64 * SLOT_BYTES,
                    &[0u8; SLOT_BYTES as usize],
                );
            }
            self.retired.push((frame_number, slot, draw));
            table_written = true;
        }

        let base = UPLOAD_BUDGET_BYTES * (frame_slot % self.frames_in_flight) as u64;
        let mut used = 0u64;
        let mut pool_copies = Vec::new();

        while let Some(upload) = self.pending.front() {
            let bytes: &[u8] = bytemuck::cast_slice(&upload.quads);
            let needed = bytes.len() as u64;
            if used + needed > UPLOAD_BUDGET_BYTES {
                break;
            }
            let Some(upload) = self.pending.pop_front() else {
                break;
            };
            let Some(draw) = self.draws[upload.slot as usize].as_mut() else {
                continue; // чанк уже убран
            };
            let bytes: &[u8] = bytemuck::cast_slice(&upload.quads);
            let offset = base + used;
            write_mapped(&mut self.staging, offset, bytes)?;
            pool_copies.push(
                vk::BufferCopy::default()
                    .src_offset(offset)
                    .dst_offset(draw.first_quad as u64 * QUAD_BYTES)
                    .size(bytes.len() as u64),
            );
            used += bytes.len() as u64;
            // SAFETY: cmd is recording outside a rendering scope; the region lies within the table.
            unsafe {
                device.cmd_update_buffer(
                    cmd,
                    self.chunk_table.buffer,
                    upload.slot as u64 * SLOT_BYTES,
                    bytemuck::cast_slice(&upload.record),
                );
            }
            table_written = true;
            draw.ready = true;
        }

        if !pool_copies.is_empty() {
            // SAFETY: cmd is recording outside a rendering scope; regions lie within the buffers.
            unsafe {
                device.cmd_copy_buffer(cmd, self.staging.buffer, self.pool.buffer, &pool_copies);
            }
        }
        if table_written || !pool_copies.is_empty() {
            record_upload_barrier(device, cmd);
        }
        Ok(used)
    }

    /// Сколько слотов проверять при отсечении (номер наибольшего занятого слота + 1).
    pub fn slot_limit(&self) -> u32 {
        self.slot_limit
    }

    /// Буфер таблицы чанков (его читает отсечение).
    pub fn chunk_table(&self) -> vk::Buffer {
        self.chunk_table.buffer
    }

    /// Записывает отрисовку по списку команд, который заполнило отсечение.
    /// `draw_count` — есть ли `vkCmdDrawIndirectCount`; без него рисуется весь используемый участок
    /// списка (неиспользованные команды обнулены и ничего не рисуют).
    pub fn record(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        push: &TerrainPush,
        list: &IndirectList,
        draw_count: bool,
    ) {
        if self.slot_limit == 0 {
            return;
        }
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        let stride = size_of::<vk::DrawIndirectCommand>() as u32;
        // SAFETY: cmd is recording inside a rendering scope with matching formats; the culling
        // barrier made the command list readable for indirect drawing.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(push),
            );
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
            if draw_count {
                device.cmd_draw_indirect_count(
                    cmd,
                    list.buffer,
                    list.offset,
                    list.count_buffer,
                    list.count_offset,
                    list.max_draws,
                    stride,
                );
            } else {
                let used = (self.slot_limit * DRAWS_PER_CHUNK).min(list.max_draws);
                device.cmd_draw_indirect(cmd, list.buffer, list.offset, used, stride);
            }
        }
    }

    pub fn stats(&self) -> TerrainStats {
        TerrainStats {
            chunks: self.live_chunks,
            quads_resident: self.resident_quads,
            pool_used_quads: self.ranges.used(),
            pool_capacity_quads: self.ranges.capacity(),
            pending_uploads: self.pending.len() as u32,
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &SharedAllocator) {
        // SAFETY: caller waited for the device to be idle.
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
        self.pool.destroy(device, allocator);
        self.chunk_table.destroy(device, allocator);
        self.colors.destroy(device, allocator);
        self.staging.destroy(device, allocator);
    }

    fn write_staging(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        write_mapped(&mut self.staging, offset, bytes)
    }
}

fn write_mapped(buffer: &mut GpuBuffer, offset: u64, bytes: &[u8]) -> Result<()> {
    let mapped = buffer.mapped().context("staging buffer is not host-visible")?;
    let start = offset as usize;
    mapped[start..start + bytes.len()].copy_from_slice(bytes);
    Ok(())
}

/// Барьер: записанное копированием должно быть видно отсечению и вершинному шейдеру
/// в этом же и следующих кадрах.
fn record_upload_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(
            vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
        )
        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ);
    // SAFETY: cmd is recording outside a rendering scope.
    unsafe {
        device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&[barrier]));
    }
}

fn create_shader(device: &Device, bytes: &[u8]) -> Result<vk::ShaderModule> {
    let code = ash::util::read_spv(&mut Cursor::new(bytes))?;
    // SAFETY: code is valid SPIR-V produced by glslc at build time.
    Ok(unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None) }?)
}

fn create_pipeline(
    device: &Device,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
    depth_format: vk::Format,
    face_culling: bool,
) -> Result<vk::Pipeline> {
    let vert = create_shader(device, TERRAIN_VERT)?;
    let frag = create_shader(device, TERRAIN_FRAG)?;
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(if face_culling {
            vk::CullModeFlags::BACK
        } else {
            vk::CullModeFlags::NONE
        })
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // Обратная глубина (reverse-Z): ближе = больше. Точнее на больших расстояниях.
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_formats = [color_format];
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(depth_format);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    // SAFETY: all referenced state lives until the call returns.
    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
    };
    // SAFETY: modules are no longer needed once the pipeline is created (or creation failed).
    unsafe {
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
    }
    let pipelines = result.map_err(|(_, error)| error).context("terrain pipeline creation failed")?;
    Ok(pipelines[0])
}

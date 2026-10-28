//! Отсечение невидимого на видеокарте.
//!
//! Из загруженных чанков рисуется только видимое. Решает это сама видеокарта (compute-шейдер
//! `cull.comp`), а процессор только запускает проверку — поэтому отсечение почти ничего
//! не стоит процессору даже при тысячах чанков.
//!
//! Три уровня (каждый выключается флагом для замеров «до/после»):
//! 1. **Пирамида видимости (frustum)** — чанк целиком вне поля зрения камеры не рисуется.
//!    Флаг `PARODIA_FRUSTUM_CULLING`.
//! 2. **Hi-Z (закрытое другим)** — чанк, целиком спрятанный за горой или стеной, не рисуется.
//!    Флаг `PARODIA_OCCLUSION_CULLING`. Работает в два прохода:
//!    - ранний проход: рисуем то, что было видно в прошлом кадре (почти всегда это почти всё);
//!    - строим пирамиду глубины (`depth_pyramid.comp`): уменьшенные копии буфера глубины,
//!      где каждая точка хранит самую дальнюю глубину своей области;
//!    - поздний проход: проверяем остальные чанки по пирамиде и дорисовываем открывшиеся.
//!      Заодно запоминаем, что видно, для следующего кадра.
//! 3. **Группы граней** — квады чанка лежат группами по сторонам; стороны, повёрнутые от камеры
//!    (например, низ чанка, когда камера над ним), не рисуются вовсе. Флаг `PARODIA_FACE_CULLING`.
//!
//! Результат — список команд indirect-отрисовки (`VkDrawIndirectCommand`) и их число;
//! ландшафт рисуется одним вызовом `vkCmdDrawIndirectCount` на проход.
//! Счётчики (сколько в кадре, сколько закрыто, сколько нарисовано) копируются в память,
//! которую читает процессор, и показываются в F3 с задержкой в пару кадров.

use std::io::Cursor;

use anyhow::{Context as _, Result};
use ash::{Device, vk};
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::resources::{GpuBuffer, GpuImage, SharedAllocator};
use crate::terrain::MAX_CHUNK_SLOTS;

const CULL_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cull.comp.spv"));
const PYRAMID_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/depth_pyramid.comp.spv"));

/// Команд отрисовки на чанк не больше 3: из 6 групп сторон видимые идут не более чем тремя кусками.
pub const DRAWS_PER_CHUNK: u32 = 3;
/// Вместимость одного списка команд (ранний и поздний проход — по списку).
pub const MAX_DRAWS: u32 = MAX_CHUNK_SLOTS * DRAWS_PER_CHUNK;
const DRAW_BYTES: u64 = 16;
/// Размер группы потоков в `cull.comp`.
const CULL_GROUP: u32 = 64;
/// Размер группы потоков в `depth_pyramid.comp` (по каждой оси).
const PYRAMID_GROUP: u32 = 8;
/// Больше уровней пирамиды не бывает (картинка до 32768 точек).
const MAX_PYRAMID_LEVELS: u32 = 16;

// Номера счётчиков — как в `cull.comp`.
const C_EARLY_DRAWS: usize = 0;
const C_LATE_DRAWS: usize = 1;
const C_IN_FRUSTUM: usize = 2;
const C_OCCLUDED: usize = 3;
const C_DRAWN_CHUNKS: usize = 4;
const C_DRAWN_QUADS: usize = 5;
const C_OVERFLOW: usize = 6;
const COUNTERS: usize = 8;
const COUNTER_BYTES: u64 = (COUNTERS * 4) as u64;

const FLAG_FRUSTUM: u32 = 1;
const FLAG_OCCLUSION: u32 = 2;
const FLAG_FACES: u32 = 4;

/// Какие уровни отсечения включены.
#[derive(Clone, Copy, Debug)]
pub struct CullSettings {
    pub frustum: bool,
    pub occlusion: bool,
    pub faces: bool,
}

/// Итог отсечения за кадр (для F3 и журнала).
#[derive(Clone, Copy, Debug, Default)]
pub struct CullStats {
    /// Чанков в поле зрения камеры.
    pub in_frustum: u32,
    /// Чанков в поле зрения, но закрытых другими (не нарисованы).
    pub occluded: u32,
    /// Чанков нарисовано.
    pub drawn_chunks: u32,
    /// Команд отрисовки (оба прохода).
    pub draws: u32,
    /// Квадов нарисовано.
    pub quads: u64,
}

/// Проход отсечения.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CullPass {
    /// Чанки, видимые в прошлом кадре.
    Early = 0,
    /// Остальные: проверка по пирамиде глубины.
    Late = 1,
}

/// Где лежат команды отрисовки одного прохода.
#[derive(Clone, Copy, Debug)]
pub struct IndirectList {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub count_buffer: vk::Buffer,
    pub count_offset: u64,
    pub max_draws: u32,
}

/// Данные камеры для отсечения (128 байт push constants, как в `cull.comp`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullPush {
    view_proj: [f32; 16],
    camera_block: [i32; 4],
    camera_frac: [f32; 4],
    /// slot count, flags, pass, max commands per list
    params: [u32; 4],
    /// pyramid width, height, mip count
    pyramid: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PyramidPush {
    src_size: [i32; 2],
    dst_size: [i32; 2],
}

/// Камера для отсечения (те же данные, что получает шейдер ландшафта).
#[derive(Clone, Copy, Debug)]
pub struct CullView {
    pub view_proj: [f32; 16],
    pub camera_block: [i32; 3],
    pub camera_frac: [f32; 3],
}

struct Pyramid {
    image: GpuImage,
    /// По одному виду на уровень — для записи уровня шейдером.
    mip_views: Vec<vk::ImageView>,
    width: u32,
    height: u32,
    levels: u32,
}

pub struct Culling {
    settings: CullSettings,
    /// Есть `vkCmdDrawIndirectCount`; без него рисуется весь список, лишние команды обнулены.
    draw_count: bool,
    /// Можно ли читать буфер глубины в шейдере (без этого Hi-Z выключен).
    occlusion_available: bool,
    /// 1 = чанк был виден в прошлом кадре (по слоту).
    visibility: GpuBuffer,
    /// Два списка команд подряд: ранний проход, затем поздний.
    draws: GpuBuffer,
    counters: GpuBuffer,
    /// Копия счётчиков для процессора: по участку на каждый кадр «в полёте».
    readback: GpuBuffer,
    readback_ready: Vec<bool>,
    /// Буфер видимости ещё не обнулён (содержимое новой видеопамяти не определено).
    needs_clear: bool,
    sampler: vk::Sampler,
    cull_set_layout: vk::DescriptorSetLayout,
    cull_pool: vk::DescriptorPool,
    cull_set: vk::DescriptorSet,
    cull_layout: vk::PipelineLayout,
    cull_pipeline: vk::Pipeline,
    pyramid_set_layout: vk::DescriptorSetLayout,
    pyramid_pool: vk::DescriptorPool,
    pyramid_sets: Vec<vk::DescriptorSet>,
    pyramid_layout: vk::PipelineLayout,
    pyramid_pipeline: vk::Pipeline,
    pyramid: Option<Pyramid>,
}

impl Culling {
    pub fn new(
        device: &Device,
        allocator: &SharedAllocator,
        chunk_table: vk::Buffer,
        settings: CullSettings,
        draw_count: bool,
        occlusion_available: bool,
        frames_in_flight: usize,
    ) -> Result<Self> {
        let rw = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let indirect = rw | vk::BufferUsageFlags::INDIRECT_BUFFER;
        let visibility = GpuBuffer::new(
            device,
            allocator,
            MAX_CHUNK_SLOTS as u64 * 4,
            rw,
            MemoryLocation::GpuOnly,
            "chunk visibility",
        )?;
        let draws = GpuBuffer::new(
            device,
            allocator,
            2 * MAX_DRAWS as u64 * DRAW_BYTES,
            indirect,
            MemoryLocation::GpuOnly,
            "terrain draw commands",
        )?;
        let counters = GpuBuffer::new(
            device,
            allocator,
            COUNTER_BYTES,
            indirect | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
            "cull counters",
        )?;
        let readback = GpuBuffer::new(
            device,
            allocator,
            COUNTER_BYTES * frames_in_flight as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "cull counters readback",
        )?;

        // SAFETY: valid device; every object below is destroyed in `destroy`.
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
        }?;

        // Отсечение: 4 буфера + пирамида глубины.
        let mut cull_bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..4)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        cull_bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
        // SAFETY: valid device.
        let cull_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&cull_bindings),
                None,
            )
        }?;
        let cull_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 4,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            },
        ];
        // SAFETY: valid device.
        let cull_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&cull_sizes),
                None,
            )
        }?;
        let cull_layouts = [cull_set_layout];
        // SAFETY: the pool has room for this set.
        let cull_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(cull_pool)
                    .set_layouts(&cull_layouts),
            )
        }?[0];
        let buffer_infos = [
            [vk::DescriptorBufferInfo::default().buffer(chunk_table).range(vk::WHOLE_SIZE)],
            [vk::DescriptorBufferInfo::default().buffer(visibility.buffer).range(vk::WHOLE_SIZE)],
            [vk::DescriptorBufferInfo::default().buffer(draws.buffer).range(vk::WHOLE_SIZE)],
            [vk::DescriptorBufferInfo::default().buffer(counters.buffer).range(vk::WHOLE_SIZE)],
        ];
        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(cull_set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        // SAFETY: the set and buffers are valid; the set is not in use yet.
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let cull_push = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: size_of::<CullPush>() as u32,
        }];
        // SAFETY: valid device and set layout.
        let cull_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&cull_layouts)
                    .push_constant_ranges(&cull_push),
                None,
            )
        }?;
        let cull_pipeline = create_compute_pipeline(device, cull_layout, CULL_COMP, "cull")?;

        // Пирамида глубины: источник (глубина или прошлый уровень) + уровень для записи.
        let pyramid_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        // SAFETY: valid device.
        let pyramid_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&pyramid_bindings),
                None,
            )
        }?;
        let pyramid_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: MAX_PYRAMID_LEVELS,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: MAX_PYRAMID_LEVELS,
            },
        ];
        // SAFETY: valid device.
        let pyramid_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(MAX_PYRAMID_LEVELS)
                    .pool_sizes(&pyramid_sizes),
                None,
            )
        }?;
        let pyramid_layouts = vec![pyramid_set_layout; MAX_PYRAMID_LEVELS as usize];
        // SAFETY: the pool has room for exactly these sets.
        let pyramid_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pyramid_pool)
                    .set_layouts(&pyramid_layouts),
            )
        }?;
        let pyramid_push = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: size_of::<PyramidPush>() as u32,
        }];
        let single_layout = [pyramid_set_layout];
        // SAFETY: valid device and set layout.
        let pyramid_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&single_layout)
                    .push_constant_ranges(&pyramid_push),
                None,
            )
        }?;
        let pyramid_pipeline =
            create_compute_pipeline(device, pyramid_layout, PYRAMID_COMP, "depth pyramid")?;

        Ok(Self {
            settings: CullSettings {
                occlusion: settings.occlusion && occlusion_available,
                ..settings
            },
            draw_count,
            occlusion_available,
            visibility,
            draws,
            counters,
            readback,
            readback_ready: vec![false; frames_in_flight],
            needs_clear: true,
            sampler,
            cull_set_layout,
            cull_pool,
            cull_set,
            cull_layout,
            cull_pipeline,
            pyramid_set_layout,
            pyramid_pool,
            pyramid_sets,
            pyramid_layout,
            pyramid_pipeline,
            pyramid: None,
        })
    }

    /// Пересоздаёт пирамиду глубины под новый размер картинки. Вызывать, когда видеокарта простаивает.
    pub fn resize(
        &mut self,
        device: &Device,
        allocator: &SharedAllocator,
        depth_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<()> {
        self.destroy_pyramid(device, allocator);

        // Уровень 0 — ближайшая степень двойки не больше экрана: дальше каждый уровень ровно вдвое меньше.
        let width = prev_power_of_two(extent.width);
        let height = prev_power_of_two(extent.height);
        let levels = (32 - width.max(height).leading_zeros()).min(MAX_PYRAMID_LEVELS);
        let image = GpuImage::new_2d(
            device,
            allocator,
            vk::Format::R32_SFLOAT,
            vk::Extent2D { width, height },
            levels,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "depth pyramid",
        )?;
        let mut mip_views = Vec::with_capacity(levels as usize);
        for level in 0..levels {
            let info = vk::ImageViewCreateInfo::default()
                .image(image.image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: level,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            // SAFETY: the image is bound to memory; views are destroyed with the pyramid.
            mip_views.push(unsafe { device.create_image_view(&info, None) }?);
        }

        let full = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(image.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(self.cull_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&full),
        ];
        let depth_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(depth_view)
            .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)];
        // Уровень N читает уровень N−1 через свой вид (только этот уровень), а пишет в свой.
        let sampled_infos: Vec<[vk::DescriptorImageInfo; 1]> = mip_views
            .iter()
            .map(|&view| {
                [vk::DescriptorImageInfo::default()
                    .sampler(self.sampler)
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)]
            })
            .collect();
        let storage_infos: Vec<[vk::DescriptorImageInfo; 1]> = mip_views
            .iter()
            .map(|&view| {
                [vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)]
            })
            .collect();
        if self.occlusion_available {
            for level in 0..levels as usize {
                let source = if level == 0 {
                    &depth_info
                } else {
                    &sampled_infos[level - 1]
                };
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.pyramid_sets[level])
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(source),
                );
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.pyramid_sets[level])
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&storage_infos[level]),
                );
            }
        }
        // SAFETY: the GPU is idle (swapchain rebuild), so the sets are not in use.
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        self.pyramid = Some(Pyramid {
            image,
            mip_views,
            width,
            height,
            levels,
        });
        Ok(())
    }

    /// Начало кадра: обнуляет счётчики, готовит пирамиду. Вызывать до `cull`, вне отрисовки.
    /// `slot_limit` — слотов чанков в работе (номер наибольшего занятого + 1).
    pub fn begin_frame(&mut self, device: &Device, cmd: vk::CommandBuffer, slot_limit: u32) {
        let Some(pyramid) = self.pyramid.as_ref() else {
            return;
        };
        // Прошлые кадры могли ещё читать и писать эти буферы — дожидаемся их.
        let previous = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::ALL_TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            )
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
        // Содержимое пирамиды с прошлого кадра не нужно — она строится заново.
        let pyramid_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(pyramid.image.image)
            .subresource_range(color_range(pyramid.levels));
        let filled = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
        // SAFETY: cmd is recording outside a rendering scope; all objects are alive.
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .memory_barriers(&[previous])
                    .image_memory_barriers(&[pyramid_barrier]),
            );
            if self.needs_clear {
                device.cmd_fill_buffer(cmd, self.visibility.buffer, 0, vk::WHOLE_SIZE, 0);
                self.needs_clear = false;
            }
            device.cmd_fill_buffer(cmd, self.counters.buffer, 0, COUNTER_BYTES, 0);
            if !self.draw_count {
                // Без vkCmdDrawIndirectCount рисуется весь список: неиспользованные команды — пустые.
                let bytes = (slot_limit.min(MAX_CHUNK_SLOTS) * DRAWS_PER_CHUNK) as u64 * DRAW_BYTES;
                if bytes > 0 {
                    for list in 0..2u64 {
                        let offset = list * MAX_DRAWS as u64 * DRAW_BYTES;
                        device.cmd_fill_buffer(cmd, self.draws.buffer, offset, bytes, 0);
                    }
                }
            }
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[filled]),
            );
        }
    }

    /// Запускает проход отсечения. Вызывать вне отрисовки; после него список команд готов.
    pub fn cull(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        pass: CullPass,
        view: &CullView,
        slot_limit: u32,
    ) {
        let Some(pyramid) = self.pyramid.as_ref() else {
            return;
        };
        if slot_limit == 0 {
            return;
        }
        let mut flags = 0;
        if self.settings.frustum {
            flags |= FLAG_FRUSTUM;
        }
        if self.settings.occlusion {
            flags |= FLAG_OCCLUSION;
        }
        if self.settings.faces {
            flags |= FLAG_FACES;
        }
        let push = CullPush {
            view_proj: view.view_proj,
            camera_block: [view.camera_block[0], view.camera_block[1], view.camera_block[2], 0],
            camera_frac: [view.camera_frac[0], view.camera_frac[1], view.camera_frac[2], 0.0],
            params: [slot_limit, flags, pass as u32, MAX_DRAWS],
            pyramid: [
                pyramid.width as f32,
                pyramid.height as f32,
                pyramid.levels as f32,
                0.0,
            ],
        };
        let done = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::DRAW_INDIRECT
                    | vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::ALL_TRANSFER,
            )
            .dst_access_mask(
                vk::AccessFlags2::INDIRECT_COMMAND_READ
                    | vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE
                    | vk::AccessFlags2::TRANSFER_READ,
            );
        // SAFETY: cmd is recording outside a rendering scope; descriptors point to live objects.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.cull_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.cull_layout,
                0,
                &[self.cull_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.cull_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_dispatch(cmd, slot_limit.div_ceil(CULL_GROUP), 1, 1);
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&[done]));
        }
    }

    /// Нужен ли поздний проход (Hi-Z включён).
    pub fn occlusion(&self) -> bool {
        self.settings.occlusion && self.pyramid.is_some()
    }

    /// Строит пирамиду глубины из буфера глубины раннего прохода.
    /// До: глубина в `DEPTH_ATTACHMENT_OPTIMAL`; после: в `DEPTH_READ_ONLY_OPTIMAL`
    /// (вернуть обратно — `restore_depth`).
    pub fn build_pyramid(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        depth_image: vk::Image,
        depth_extent: vk::Extent2D,
    ) {
        let Some(pyramid) = self.pyramid.as_ref() else {
            return;
        };
        let to_read = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
            .image(depth_image)
            .subresource_range(depth_only_range());
        // Ранняя отрисовка читала счётчики и команды — поздний проход перезапишет соседние участки.
        let indirect_done = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::NONE);
        let level_done = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ);
        // SAFETY: cmd is recording outside a rendering scope; descriptors point to live objects.
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .memory_barriers(&[indirect_done])
                    .image_memory_barriers(&[to_read]),
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pyramid_pipeline);
            let mut src_size = [depth_extent.width as i32, depth_extent.height as i32];
            for level in 0..pyramid.levels {
                let dst_size = [
                    (pyramid.width >> level).max(1) as i32,
                    (pyramid.height >> level).max(1) as i32,
                ];
                let push = PyramidPush { src_size, dst_size };
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pyramid_layout,
                    0,
                    &[self.pyramid_sets[level as usize]],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    self.pyramid_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_dispatch(
                    cmd,
                    (dst_size[0] as u32).div_ceil(PYRAMID_GROUP),
                    (dst_size[1] as u32).div_ceil(PYRAMID_GROUP),
                    1,
                );
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&[level_done]),
                );
                src_size = dst_size;
            }
        }
    }

    /// Возвращает глубину в режим «рисуем в неё» (содержимое сохраняется).
    pub fn restore_depth(&self, device: &Device, cmd: vk::CommandBuffer, depth_image: vk::Image) {
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .image(depth_image)
            .subresource_range(depth_only_range());
        // SAFETY: cmd is recording outside a rendering scope.
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[barrier]),
            );
        }
    }

    /// Список команд прохода.
    pub fn list(&self, pass: CullPass) -> IndirectList {
        let (offset, count) = match pass {
            CullPass::Early => (0, C_EARLY_DRAWS),
            CullPass::Late => (MAX_DRAWS as u64 * DRAW_BYTES, C_LATE_DRAWS),
        };
        IndirectList {
            buffer: self.draws.buffer,
            offset,
            count_buffer: self.counters.buffer,
            count_offset: (count * 4) as u64,
            max_draws: MAX_DRAWS,
        }
    }

    /// Есть ли `vkCmdDrawIndirectCount`.
    pub fn draw_count_supported(&self) -> bool {
        self.draw_count
    }

    /// Копирует счётчики кадра в память для процессора. Вызывать после последнего прохода отсечения.
    pub fn copy_stats(&mut self, device: &Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        let region = [vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(frame_slot as u64 * COUNTER_BYTES)
            .size(COUNTER_BYTES)];
        let to_host = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ);
        // SAFETY: cmd is recording outside a rendering scope; the cull barrier made the counters readable.
        unsafe {
            device.cmd_copy_buffer(cmd, self.counters.buffer, self.readback.buffer, &region);
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&[to_host]));
        }
        if let Some(ready) = self.readback_ready.get_mut(frame_slot) {
            *ready = true;
        }
    }

    /// Счётчики кадра, который последним рисовался в слоте `frame_slot`.
    /// Вызывать после ожидания fence этого слота.
    pub fn read_stats(&mut self, frame_slot: usize) -> Option<CullStats> {
        if !self.readback_ready.get(frame_slot).copied().unwrap_or(false) {
            return None;
        }
        let mapped = self.readback.mapped()?;
        let start = frame_slot * COUNTER_BYTES as usize;
        let bytes = mapped.get(start..start + COUNTER_BYTES as usize)?;
        let mut c = [0u32; COUNTERS];
        for (value, chunk) in c.iter_mut().zip(bytes.chunks_exact(4)) {
            *value = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        if c[C_OVERFLOW] > 0 {
            log::warn!("terrain draw list overflow: {} commands dropped", c[C_OVERFLOW]);
        }
        Some(CullStats {
            in_frustum: c[C_IN_FRUSTUM],
            occluded: c[C_OCCLUDED],
            drawn_chunks: c[C_DRAWN_CHUNKS],
            draws: c[C_EARLY_DRAWS].min(MAX_DRAWS) + c[C_LATE_DRAWS].min(MAX_DRAWS),
            quads: c[C_DRAWN_QUADS] as u64,
        })
    }

    fn destroy_pyramid(&mut self, device: &Device, allocator: &SharedAllocator) {
        if let Some(mut pyramid) = self.pyramid.take() {
            // SAFETY: caller guarantees the GPU no longer uses the pyramid.
            unsafe {
                for view in pyramid.mip_views.drain(..) {
                    device.destroy_image_view(view, None);
                }
            }
            pyramid.image.destroy(device, allocator);
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &SharedAllocator) {
        self.destroy_pyramid(device, allocator);
        // SAFETY: caller waited for the device to be idle.
        unsafe {
            device.destroy_pipeline(self.cull_pipeline, None);
            device.destroy_pipeline_layout(self.cull_layout, None);
            device.destroy_descriptor_pool(self.cull_pool, None);
            device.destroy_descriptor_set_layout(self.cull_set_layout, None);
            device.destroy_pipeline(self.pyramid_pipeline, None);
            device.destroy_pipeline_layout(self.pyramid_layout, None);
            device.destroy_descriptor_pool(self.pyramid_pool, None);
            device.destroy_descriptor_set_layout(self.pyramid_set_layout, None);
            device.destroy_sampler(self.sampler, None);
        }
        self.visibility.destroy(device, allocator);
        self.draws.destroy(device, allocator);
        self.counters.destroy(device, allocator);
        self.readback.destroy(device, allocator);
    }
}

/// Наибольшая степень двойки, не превышающая `value` (для 0 — 1).
fn prev_power_of_two(value: u32) -> u32 {
    if value <= 1 {
        1
    } else {
        1 << (31 - value.leading_zeros())
    }
}

fn color_range(levels: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: levels,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn depth_only_range() -> vk::ImageSubresourceRange {
    crate::resources::depth_range()
}

fn create_compute_pipeline(
    device: &Device,
    layout: vk::PipelineLayout,
    spirv: &[u8],
    name: &str,
) -> Result<vk::Pipeline> {
    let code = ash::util::read_spv(&mut Cursor::new(spirv))?;
    // SAFETY: code is valid SPIR-V produced by glslc at build time.
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
    }?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    // SAFETY: all referenced state lives until the call returns.
    let result =
        unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
    // SAFETY: the module is no longer needed once the pipeline is created (or creation failed).
    unsafe { device.destroy_shader_module(module, None) };
    let pipelines = result
        .map_err(|(_, error)| error)
        .with_context(|| format!("{name} pipeline creation failed"))?;
    Ok(pipelines[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prev_power_of_two_rounds_down() {
        assert_eq!(prev_power_of_two(0), 1);
        assert_eq!(prev_power_of_two(1), 1);
        assert_eq!(prev_power_of_two(720), 512);
        assert_eq!(prev_power_of_two(1024), 1024);
        assert_eq!(prev_power_of_two(1920), 1024);
    }

    #[test]
    fn push_constants_fit_the_guaranteed_limit() {
        assert_eq!(size_of::<CullPush>(), 128);
        assert_eq!(size_of::<PyramidPush>(), 16);
    }
}

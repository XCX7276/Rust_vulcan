//! Главный объект рендера.
//!
//! Кадр устроен так:
//! 1. ждём, пока видеокарта закончит кадр, который рисовался в этом же «слоте» 2 кадра назад
//!    (кадров в полёте два: пока видеокарта рисует один, процессор готовит следующий);
//! 2. берём свободную картинку из swapchain;
//! 3. записываем команды: очистка цвета и глубины → ландшафт → отладочный интерфейс;
//! 4. отправляем команды видеокарте и просим показать картинку в окне.
//!
//! Тени, небо, вода и прочее будут добавляться в шаг 3 на следующих этапах.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use ash::vk;
use egui_ash_renderer::allocator::GpuAllocator;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::context::{GpuContext, GpuInfo};
use crate::resources::{GpuImage, SharedAllocator, depth_range};
use crate::swapchain::{Swapchain, color_range, is_srgb};
use crate::terrain::{ChunkMeshId, ChunkUpload, Terrain, TerrainPush, TerrainStats};
use crate::timing::GpuTimer;

/// Сколько кадров может одновременно готовиться (процессор) и рисоваться (видеокарта).
pub const FRAMES_IN_FLIGHT: usize = 2;

/// Цвет очистки экрана, когда сцены нет (линейный цвет, светло-голубое небо).
const CLEAR_COLOR: [f32; 3] = [0.30, 0.50, 0.80];
/// Формат буфера глубины (32-битное число с плавающей точкой — обязателен для всех видеокарт Vulkan).
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// Настройки при создании рендера.
pub struct RendererConfig {
    pub vsync: bool,
    pub validation: bool,
    /// Отсечение задних граней (выключается флагом `PARODIA_FACE_CULLING=0` для отладки).
    pub face_culling: bool,
    pub width: u32,
    pub height: u32,
}

/// Что нужно рендеру, чтобы нарисовать мир в этом кадре.
#[derive(Clone, Copy, Debug)]
pub struct SceneView {
    /// Матрица «поворот камеры × проекция» (камера в начале координат), по столбцам.
    pub view_proj: [f32; 16],
    /// Целая часть позиции камеры (в блоках) относительно точки отсчёта рендера.
    pub camera_block: [i32; 3],
    /// Дробная часть позиции камеры (0..1 по каждой оси).
    pub camera_frac: [f32; 3],
    /// Расстояние, на котором туман полностью скрывает мир.
    pub fog_distance: f32,
    /// Направление на солнце.
    pub sun_dir: [f32; 3],
    /// Цвет неба и тумана (линейный).
    pub sky_color: [f32; 3],
}

/// Готовый к отрисовке отладочный интерфейс (результат egui за этот кадр).
///
/// `textures_delta` — изменения текстур интерфейса. Рендер забирает из него всё, что применил;
/// если кадр пропущен, остаток остаётся в нём и должен быть передан в следующий кадр.
pub struct UiFrame<'a> {
    pub textures_delta: &'a mut egui::TexturesDelta,
    pub primitives: &'a [egui::ClippedPrimitive],
    pub pixels_per_point: f32,
}

/// Замеры одного кадра.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    /// Сколько процессор потратил на подготовку и отправку кадра (мс).
    pub cpu_ms: f32,
    /// Сколько видеокарта рисовала кадр (мс); приходит с задержкой в пару кадров.
    pub gpu_ms: Option<f32>,
    /// Сколько вызовов отрисовки ландшафта.
    pub terrain_draws: u32,
    /// Сколько квадов ландшафта отправлено на отрисовку.
    pub terrain_quads: u64,
}

/// Ресурсы одного кадра «в полёте».
struct FrameSlot {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    /// Сигнал «картинка swapchain свободна, можно рисовать».
    image_available: vk::Semaphore,
    /// Сигнал процессору «видеокарта закончила этот кадр».
    in_flight: vk::Fence,
    /// Номер кадра, который последним рисовался в этом слоте.
    frame_number: u64,
}

pub struct Renderer {
    ui: Option<egui_ash_renderer::Renderer<GpuAllocator>>,
    ui_format: vk::Format,
    terrain: Option<Terrain>,
    depth: Option<GpuImage>,
    allocator: Option<SharedAllocator>,
    upload_pool: vk::CommandPool,
    frames: Vec<FrameSlot>,
    frame_index: usize,
    /// Сквозной номер кадра (растёт на 1 каждый отправленный кадр).
    frame_number: u64,
    /// Текстуры интерфейса, которые можно удалить, когда слот снова освободится.
    retired_ui_textures: Vec<Vec<egui::TextureId>>,
    swapchain: Option<Swapchain>,
    swapchain_loader: ash::khr::swapchain::Device,
    needs_rebuild: bool,
    window_size: (u32, u32),
    vsync: bool,
    face_culling: bool,
    timer: Option<GpuTimer>,
    last_gpu_ms: Option<f32>,
    // Должен уничтожаться последним — поэтому стоит в конце структуры.
    ctx: GpuContext,
}

impl Renderer {
    pub fn new(
        display: &impl HasDisplayHandle,
        window: &impl HasWindowHandle,
        config: RendererConfig,
    ) -> Result<Self> {
        let ctx = GpuContext::new(display, window, config.validation)?;
        log::info!(
            "GPU: {} ({:?}), Vulkan {}.{}.{}, driver {}",
            ctx.info.name,
            ctx.info.device_type,
            vk::api_version_major(ctx.info.api_version),
            vk::api_version_minor(ctx.info.api_version),
            vk::api_version_patch(ctx.info.api_version),
            ctx.info.driver
        );

        // SAFETY: valid physical device.
        let depth_props = unsafe {
            ctx.instance
                .get_physical_device_format_properties(ctx.physical_device, DEPTH_FORMAT)
        };
        if !depth_props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            bail!("GPU does not support D32_SFLOAT depth buffers");
        }

        let swapchain_loader = ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device);
        let device = &ctx.device;

        let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let pool_info =
                vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family);
            // SAFETY: valid device; objects are destroyed in Drop.
            unsafe {
                let command_pool = device.create_command_pool(&pool_info, None)?;
                let alloc = vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let command_buffer = device.allocate_command_buffers(&alloc)?[0];
                let image_available =
                    device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                let in_flight = device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?;
                frames.push(FrameSlot {
                    command_pool,
                    command_buffer,
                    image_available,
                    in_flight,
                    frame_number: 0,
                });
            }
        }

        // Отдельный пул команд для загрузок (текстуры интерфейса, меши чанков).
        let upload_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: valid device.
        let upload_pool = unsafe { device.create_command_pool(&upload_info, None) }?;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: ctx.instance.clone(),
            device: ctx.device.clone(),
            physical_device: ctx.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .context("failed to create GPU memory allocator")?;

        let timer = if ctx.info.timestamps_supported {
            Some(GpuTimer::new(device, FRAMES_IN_FLIGHT, ctx.info.timestamp_period_ns)?)
        } else {
            log::warn!("GPU timestamps not supported; GPU frame time will not be shown");
            None
        };

        let mut renderer = Self {
            ui: None,
            ui_format: vk::Format::UNDEFINED,
            terrain: None,
            depth: None,
            allocator: Some(Arc::new(Mutex::new(allocator))),
            upload_pool,
            frames,
            frame_index: 0,
            frame_number: 0,
            retired_ui_textures: vec![Vec::new(); FRAMES_IN_FLIGHT],
            swapchain: None,
            swapchain_loader,
            needs_rebuild: true,
            window_size: (config.width, config.height),
            vsync: config.vsync,
            face_culling: config.face_culling,
            timer,
            last_gpu_ms: None,
            ctx,
        };
        renderer.rebuild_swapchain()?;
        Ok(renderer)
    }

    pub fn gpu_info(&self) -> &GpuInfo {
        &self.ctx.info
    }

    pub fn validation_enabled(&self) -> bool {
        self.ctx.validation
    }

    /// Текущий размер картинки (0×0, если окно свёрнуто).
    pub fn extent(&self) -> (u32, u32) {
        self.swapchain
            .as_ref()
            .map_or((0, 0), |s| (s.extent.width, s.extent.height))
    }

    pub fn present_mode(&self) -> Option<vk::PresentModeKHR> {
        self.swapchain.as_ref().map(|s| s.present_mode)
    }

    /// Счётчики ландшафта для экрана F3.
    pub fn terrain_stats(&self) -> TerrainStats {
        self.terrain.as_ref().map(Terrain::stats).unwrap_or_default()
    }

    /// Окно изменило размер — swapchain пересоздастся перед следующим кадром.
    pub fn resize(&mut self, width: u32, height: u32) {
        // Окно присылает событие размера и при запуске — не пересоздаём, если размер не изменился.
        if (width, height) == self.window_size && (width, height) == self.extent() {
            return;
        }
        self.window_size = (width, height);
        self.needs_rebuild = true;
    }

    /// Задаёт цвета блоков (линейные RGBA по номеру блока). TEMPORARY: до появления текстур.
    pub fn set_block_colors(&mut self, colors: &[[f32; 4]]) -> Result<()> {
        let terrain = self.terrain.as_mut().context("terrain renderer is not ready")?;
        terrain.set_block_colors(&self.ctx.device, self.ctx.queue, self.upload_pool, colors)
    }

    /// Загружает меши чанков в видеопамять.
    pub fn upload_chunks(&mut self, chunks: &[ChunkUpload<'_>]) -> Result<Vec<Option<ChunkMeshId>>> {
        let terrain = self.terrain.as_mut().context("terrain renderer is not ready")?;
        terrain.upload(&self.ctx.device, self.ctx.queue, self.upload_pool, chunks)
    }

    /// Убирает меш чанка (место освободится, когда видеокарта закончит текущие кадры).
    pub fn remove_chunk(&mut self, id: ChunkMeshId) {
        if let Some(terrain) = self.terrain.as_mut() {
            terrain.remove(id, self.frame_number);
        }
    }

    /// Рисует и показывает один кадр. `Ok(None)` — кадр пропущен (окно свёрнуто или пересоздаётся swapchain).
    pub fn render(
        &mut self,
        scene: Option<&SceneView>,
        mut ui: Option<UiFrame<'_>>,
    ) -> Result<Option<FrameStats>> {
        let started = Instant::now();
        if self.window_size.0 == 0 || self.window_size.1 == 0 {
            return Ok(None);
        }
        if self.needs_rebuild {
            self.rebuild_swapchain()?;
        }
        let Some(swapchain) = self.swapchain.as_ref() else {
            return Ok(None);
        };
        let Some(depth) = self.depth.as_ref() else {
            return Ok(None);
        };

        let device = &self.ctx.device;
        let slot = self.frame_index;
        let frame = &self.frames[slot];

        // 1. Ждём завершения прошлого кадра этого слота.
        // SAFETY: fence belongs to this device.
        unsafe { device.wait_for_fences(&[frame.in_flight], true, u64::MAX) }?;
        if let Some(timer) = &self.timer
            && let Some(ms) = timer.read(device, slot)
        {
            self.last_gpu_ms = Some(ms);
        }
        // Кадр этого слота (и все более ранние) завершён — освобождаем то, что ждало его окончания.
        if let Some(terrain) = self.terrain.as_mut() {
            terrain.release_retired(frame.frame_number);
        }

        // Удаляем текстуры интерфейса, которые этот слот больше не использует.
        let retired = std::mem::take(&mut self.retired_ui_textures[slot]);
        if let Some(ui_renderer) = self.ui.as_mut() {
            for id in retired {
                ui_renderer.free_texture(id)?;
            }
            // Новые и изменённые текстуры интерфейса (например, шрифт) загружаем до записи кадра.
            if let Some(ui) = ui.as_mut() {
                for (id, deltas) in ui.textures_delta.set.drain() {
                    for delta in &deltas {
                        ui_renderer.set_texture(self.ctx.queue, self.upload_pool, id, delta)?;
                    }
                }
            }
        }

        // 2. Берём свободную картинку swapchain.
        // SAFETY: swapchain and semaphore are valid; the semaphore is unsignaled (its last wait completed).
        let acquired = unsafe {
            self.swapchain_loader.acquire_next_image(
                swapchain.handle,
                u64::MAX,
                frame.image_available,
                vk::Fence::null(),
            )
        };
        let (image_index, suboptimal) = match acquired {
            Ok(result) => result,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.needs_rebuild = true;
                return Ok(None);
            }
            Err(error) => return Err(error).context("vkAcquireNextImageKHR failed"),
        };
        let image = swapchain.images[image_index as usize];
        let view = swapchain.views[image_index as usize];
        let render_finished = swapchain.render_finished[image_index as usize];
        let extent = swapchain.extent;
        let cmd = frame.command_buffer;
        let in_flight = frame.in_flight;
        let image_available = frame.image_available;

        // 3. Записываем команды кадра.
        // SAFETY: the slot's fence was waited on, so its pool and command buffer are free.
        unsafe {
            device.reset_fences(&[in_flight])?;
            device.reset_command_pool(frame.command_pool, vk::CommandPoolResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        if let Some(timer) = self.timer.as_mut() {
            timer.begin(device, cmd, slot);
        }

        // Картинка: «неизвестное содержимое» → «в неё рисуем».
        let to_attachment = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(color_range());
        // Глубина одна на все кадры: ждём, пока прошлый кадр закончит с ней работать.
        let depth_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .image(depth.image)
            .subresource_range(depth_range());
        let sky = scene.map_or(CLEAR_COLOR, |s| s.sky_color);
        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [sky[0], sky[1], sky[2], 1.0],
            },
        };
        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear)];
        // Обратная глубина: очищаем нулём («бесконечно далеко»).
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(depth.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);
        // Картинка: «в неё рисовали» → «готова к показу».
        let to_present = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(image)
            .subresource_range(color_range());

        // SAFETY: cmd is recording; all referenced objects are alive for this frame.
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_attachment, depth_barrier]),
            );
            device.cmd_begin_rendering(cmd, &rendering);
        }

        let (mut terrain_draws, mut terrain_quads) = (0, 0);
        if let (Some(terrain), Some(scene)) = (self.terrain.as_ref(), scene) {
            let push = TerrainPush {
                view_proj: scene.view_proj,
                camera_block: [
                    scene.camera_block[0],
                    scene.camera_block[1],
                    scene.camera_block[2],
                    0,
                ],
                camera_frac: [
                    scene.camera_frac[0],
                    scene.camera_frac[1],
                    scene.camera_frac[2],
                    scene.fog_distance,
                ],
                sun_dir: [scene.sun_dir[0], scene.sun_dir[1], scene.sun_dir[2], 0.0],
                fog_color: [scene.sky_color[0], scene.sky_color[1], scene.sky_color[2], 1.0],
            };
            (terrain_draws, terrain_quads) = terrain.record(device, cmd, extent, &push);
        }

        if let (Some(ui_renderer), Some(ui)) = (self.ui.as_mut(), &ui) {
            ui_renderer.cmd_draw(cmd, extent, ui.pixels_per_point, ui.primitives)?;
        }
        // SAFETY: cmd is recording inside the rendering scope begun above.
        unsafe {
            device.cmd_end_rendering(cmd);
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_present]),
            );
        }
        if let Some(timer) = self.timer.as_mut() {
            timer.end(device, cmd, slot);
        }
        // SAFETY: cmd is recording.
        unsafe { device.end_command_buffer(cmd) }?;

        // 4. Отправляем видеокарте и показываем.
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(image_available)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(render_finished)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(&command_buffers)
            .signal_semaphore_infos(&signals);
        // SAFETY: all handles are valid; the fence was reset above.
        unsafe { device.queue_submit2(self.ctx.queue, &[submit], in_flight) }
            .context("vkQueueSubmit2 failed")?;
        self.frame_number += 1;
        self.frames[slot].frame_number = self.frame_number;
        let cpu_ms = started.elapsed().as_secs_f32() * 1000.0;

        let present_waits = [render_finished];
        let swapchains = [swapchain.handle];
        let indices = [image_index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&present_waits)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: the image was acquired this frame and its rendering was submitted.
        match unsafe { self.swapchain_loader.queue_present(self.ctx.queue, &present) } {
            Ok(present_suboptimal) => {
                if suboptimal || present_suboptimal {
                    self.needs_rebuild = true;
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.needs_rebuild = true,
            Err(error) => return Err(error).context("vkQueuePresentKHR failed"),
        }

        // Текстуры, которые egui просит удалить, удаляем, когда этот слот освободится (через 2 кадра).
        if let Some(ui) = ui.as_mut() {
            self.retired_ui_textures[slot].extend(ui.textures_delta.free.drain());
        }
        self.frame_index = (slot + 1) % FRAMES_IN_FLIGHT;

        Ok(Some(FrameStats {
            cpu_ms,
            gpu_ms: self.last_gpu_ms,
            terrain_draws,
            terrain_quads,
        }))
    }

    /// Пересоздаёт swapchain и буфер глубины под текущий размер окна.
    fn rebuild_swapchain(&mut self) -> Result<()> {
        let (width, height) = self.window_size;
        if width == 0 || height == 0 {
            return Ok(());
        }
        let Some(allocator) = self.allocator.clone() else {
            return Ok(());
        };
        // Ожидание простоя допустимо: это происходит только при смене размера окна.
        // SAFETY: valid device.
        unsafe { self.ctx.device.device_wait_idle() }?;

        let old = self.swapchain.take();
        let old_handle = old.as_ref().map_or(vk::SwapchainKHR::null(), |s| s.handle);
        let new = Swapchain::new(
            &self.ctx,
            self.swapchain_loader.clone(),
            width,
            height,
            self.vsync,
            old_handle,
        )?;
        if let Some(mut old) = old {
            old.destroy(&self.ctx.device);
        }
        if let Some(mut depth) = self.depth.take() {
            depth.destroy(&self.ctx.device, &allocator);
        }
        self.swapchain = new;
        self.needs_rebuild = false;

        let Some((format, extent)) = self.swapchain.as_ref().map(|s| (s.format.format, s.extent))
        else {
            return Ok(());
        };
        self.depth = Some(GpuImage::depth(
            &self.ctx.device,
            &allocator,
            DEPTH_FORMAT,
            extent,
        )?);

        let format_changed = self.ui_format != format;
        match self.terrain.as_mut() {
            None => {
                self.terrain = Some(Terrain::new(
                    &self.ctx.device,
                    &allocator,
                    format,
                    DEPTH_FORMAT,
                    self.face_culling,
                )?);
            }
            Some(terrain) if format_changed => {
                terrain.recreate_pipeline(&self.ctx.device, format, DEPTH_FORMAT)?;
            }
            Some(_) => {}
        }
        self.ensure_ui_renderer(format)?;
        Ok(())
    }

    /// Создаёт отрисовщик интерфейса под формат картинок (или пересоздаёт, если формат сменился).
    fn ensure_ui_renderer(&mut self, format: vk::Format) -> Result<()> {
        if self.ui.is_some() && self.ui_format == format {
            return Ok(());
        }
        self.ui = None;
        let Some(allocator) = self.allocator.clone() else {
            return Ok(());
        };
        let renderer = egui_ash_renderer::Renderer::with_gpu_allocator(
            allocator,
            self.ctx.device.clone(),
            egui_ash_renderer::RenderMode::DynamicRendering(egui_ash_renderer::DynamicRendering {
                color_attachment_format: format,
                depth_attachment_format: Some(DEPTH_FORMAT),
                stencil_attachment_format: None,
            }),
            egui_ash_renderer::Options {
                in_flight_frames: FRAMES_IN_FLIGHT,
                enable_depth_test: false,
                enable_depth_write: false,
                srgb_framebuffer: is_srgb(format),
            },
        )
        .context("failed to create debug UI renderer")?;
        self.ui = Some(renderer);
        self.ui_format = format;
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let device = &self.ctx.device;
        // SAFETY: we wait for the GPU, then destroy children before the context (dropped last).
        unsafe {
            let _ = device.device_wait_idle();
        }
        self.ui = None;
        if let Some(allocator) = self.allocator.clone() {
            if let Some(mut terrain) = self.terrain.take() {
                terrain.destroy(device, &allocator);
            }
            if let Some(mut depth) = self.depth.take() {
                depth.destroy(device, &allocator);
            }
        }
        // SAFETY: the GPU is idle; these objects are not used anymore.
        unsafe {
            if let Some(timer) = self.timer.take() {
                timer.destroy(device);
            }
            if let Some(mut swapchain) = self.swapchain.take() {
                swapchain.destroy(device);
            }
            for frame in self.frames.drain(..) {
                device.destroy_semaphore(frame.image_available, None);
                device.destroy_fence(frame.in_flight, None);
                device.destroy_command_pool(frame.command_pool, None);
            }
            device.destroy_command_pool(self.upload_pool, None);
        }
        // Распределитель памяти освобождается до устройства.
        self.allocator = None;
        log::info!("renderer destroyed");
    }
}

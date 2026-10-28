//! Приложение: окно, события и цикл кадров.
//!
//! winit сообщает о событиях (окно создано, изменён размер, нажата клавиша, мышь сдвинулась,
//! пора рисовать), а `App` на них отвечает. Каждый кадр:
//! сдвинуть камеру → собрать данные для F3 → построить интерфейс egui → отдать всё рендеру → посчитать FPS.
//!
//! Флаги окружения:
//! - `PARODIA_VSYNC=0` — без вертикальной синхронизации (для замеров FPS);
//! - `PARODIA_VALIDATION=1/0` — проверка ошибок Vulkan (по умолчанию включена в отладочной сборке);
//! - `PARODIA_LANG=en` — английский интерфейс (по умолчанию русский);
//! - `PARODIA_SEED=<число>` — зерно мира (по умолчанию 1337);
//! - `PARODIA_RADIUS=<чанков>` — радиус загрузки мира по горизонтали (по умолчанию 8);
//! - `PARODIA_VRADIUS=<чанков>` — окно загрузки по высоте вокруг камеры (по умолчанию 4);
//! - `PARODIA_WORKERS=<число>` — рабочих потоков генерации и мешинга (по умолчанию ядра − 2, не больше 6);
//! - `PARODIA_FACE_CULLING=0` — рисовать и задние грани (для отладки).

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result};
use glam::DVec3;
use parodia_core::env_flags;
use parodia_core::frame_timer::FpsCounter;
use parodia_core::jobs::default_worker_count;
use parodia_render::{FrameStats, Renderer, RendererConfig, SceneView, UiFrame, vk};
use parodia_ui::{DebugStats, GpuKind, Lang, WorldDebug};
use parodia_world::{BlockPos, BlockRegistry, ChunkPos, TerrainGenerator};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::camera::FlyCamera;
use crate::perf_log::PerfLog;
use crate::streaming::{StreamConfig, TerrainStreamer};

/// Направление на солнце (пока неподвижное; смена суток — этап 7).
const SUN_DIRECTION: [f32; 3] = [0.38, 0.85, 0.36];
/// Цвет неба и тумана (линейный).
const SKY_COLOR: [f32; 3] = [0.30, 0.50, 0.80];

/// Мир, загруженный для этого окна.
struct WorldView {
    seed: u64,
    /// Точка отсчёта рендера: координаты чанков для видеокарты считаются от неё (в 32-битных числах).
    render_origin: BlockPos,
    /// Радиус загрузки в блоках (для тумана).
    radius_blocks: f32,
    camera: FlyCamera,
    streamer: TerrainStreamer,
}

/// Всё, что существует, пока открыто окно.
/// Порядок полей важен: рендер должен уничтожаться раньше окна.
struct Running {
    renderer: Renderer,
    egui_state: egui_winit::State,
    egui_ctx: egui::Context,
    world: WorldView,
    window: Arc<Window>,
}

pub struct App {
    lang: Lang,
    vsync: bool,
    validation: bool,
    face_culling: bool,
    running: Option<Running>,
    fps: FpsCounter,
    last_frame: FrameStats,
    last_frame_at: Option<Instant>,
    show_debug: bool,
    keys: HashSet<KeyCode>,
    mouse_captured: bool,
    /// Изменения текстур интерфейса, ещё не переданные рендеру (копятся, если кадр пропущен).
    pending_textures: egui::TexturesDelta,
    perf_log: PerfLog,
    error: Option<anyhow::Error>,
}

impl App {
    pub fn new() -> Result<Self> {
        let lang_code = env_flags::text("LANG").unwrap_or_else(|| "ru".into());
        let lang = Lang::load(&lang_code)?;
        let vsync = env_flags::flag("VSYNC", true);
        let validation = env_flags::flag("VALIDATION", cfg!(debug_assertions));
        let face_culling = env_flags::flag("FACE_CULLING", true);
        log::info!(
            "settings: lang={}, vsync={vsync}, validation={validation}, face_culling={face_culling}",
            lang.code()
        );
        Ok(Self {
            lang,
            vsync,
            validation,
            face_culling,
            running: None,
            fps: FpsCounter::new(),
            last_frame: FrameStats::default(),
            last_frame_at: None,
            show_debug: true,
            keys: HashSet::new(),
            mouse_captured: false,
            pending_textures: egui::TexturesDelta::default(),
            perf_log: PerfLog::new(),
            error: None,
        })
    }

    /// Ошибка, из-за которой игра остановилась (если была).
    pub fn take_error(&mut self) -> Option<anyhow::Error> {
        self.error.take()
    }

    fn start(&self, event_loop: &ActiveEventLoop) -> Result<Running> {
        let attributes = Window::default_attributes()
            .with_title("Parodia")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .context("failed to create window")?,
        );
        let size = window.inner_size();
        let mut renderer = Renderer::new(
            window.as_ref(),
            window.as_ref(),
            RendererConfig {
                vsync: self.vsync,
                validation: self.validation,
                face_culling: self.face_culling,
                width: size.width,
                height: size.height,
            },
        )?;

        let world = load_world(&mut renderer)?;

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        Ok(Running {
            renderer,
            egui_state,
            egui_ctx,
            world,
            window,
        })
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: anyhow::Error) {
        log::error!("{error:#}");
        self.error = Some(error);
        self.pending_textures.clear();
        self.running = None;
        event_loop.exit();
    }

    fn set_mouse_captured(&mut self, captured: bool) {
        let Some(run) = self.running.as_ref() else {
            return;
        };
        if captured {
            // На Windows «Locked» не поддерживается — тогда держим курсор внутри окна («Confined»).
            let grabbed = run
                .window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| run.window.set_cursor_grab(CursorGrabMode::Confined));
            if let Err(error) = grabbed {
                log::warn!("cursor grab failed: {error}");
                return;
            }
            run.window.set_cursor_visible(false);
        } else {
            let _ = run.window.set_cursor_grab(CursorGrabMode::None);
            run.window.set_cursor_visible(true);
        }
        self.mouse_captured = captured;
    }

    fn redraw(&mut self) -> Result<()> {
        let now = Instant::now();
        let dt = self
            .last_frame_at
            .map_or(0.0, |t| (now - t).as_secs_f64())
            .min(0.1);
        self.last_frame_at = Some(now);

        let Some(run) = self.running.as_mut() else {
            return Ok(());
        };

        if self.mouse_captured {
            run.world.camera.update(dt, &self.keys);
        }
        let stream_started = Instant::now();
        run.world.streamer.update(
            &mut run.renderer,
            run.world.camera.position,
            run.world.camera.forward(),
        );
        let stream_ms = stream_started.elapsed().as_secs_f32() * 1000.0;

        let (width, height) = run.renderer.extent();
        let aspect = if height > 0 {
            width as f32 / height as f32
        } else {
            1.0
        };
        let camera = &run.world.camera;
        let (block, frac) = camera.split_position();
        let origin = run.world.render_origin;
        let scene = SceneView {
            view_proj: camera.view_proj(aspect).to_cols_array(),
            camera_block: [
                (block[0] - origin.x) as i32,
                (block[1] - origin.y) as i32,
                (block[2] - origin.z) as i32,
            ],
            camera_frac: frac,
            fog_distance: run.world.radius_blocks,
            sun_dir: SUN_DIRECTION,
            sky_color: SKY_COLOR,
        };

        let stats = collect_debug_stats(
            &run.renderer,
            &self.fps,
            &self.last_frame,
            &run.world,
            stream_ms,
            self.mouse_captured,
        );
        let show_debug = self.show_debug;
        let lang = &self.lang;

        let raw_input = run.egui_state.take_egui_input(&run.window);
        let output = run.egui_ctx.run_ui(raw_input, |ui| {
            if show_debug {
                parodia_ui::debug_overlay(ui.ctx(), lang, &stats);
            }
        });
        run.egui_state
            .handle_platform_output(&run.window, output.platform_output);
        let primitives = run
            .egui_ctx
            .tessellate(output.shapes, output.pixels_per_point);

        self.pending_textures.append(output.textures_delta);
        let ui = UiFrame {
            textures_delta: &mut self.pending_textures,
            primitives: &primitives,
            pixels_per_point: output.pixels_per_point,
        };
        if let Some(frame) = run.renderer.render(Some(&scene), Some(ui))? {
            self.last_frame = frame;
            self.fps.tick();
            self.perf_log.add(&frame, width, height);
        }
        Ok(())
    }
}

/// Создаёт мир: реестр блоков, генератор, стример чанков, камера в точке появления.
/// Сам ландшафт подгружается в фоне, начиная с первого кадра.
fn load_world(renderer: &mut Renderer) -> Result<WorldView> {
    let registry = BlockRegistry::builtin()?;
    // Цвета из файла — в sRGB; шейдер работает в линейном цвете.
    let colors: Vec<[f32; 4]> = registry
        .iter()
        .map(|(_, def)| {
            let [r, g, b] = def.color;
            [r.powf(2.2), g.powf(2.2), b.powf(2.2), 1.0]
        })
        .collect();
    renderer.set_block_colors(&colors)?;

    let number = |name: &str, default: i64, min: i64, max: i64| {
        env_flags::text(name)
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(default)
            .clamp(min, max)
    };
    let seed = env_flags::text("SEED")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(1337);
    let config = StreamConfig {
        radius: number("RADIUS", 8, 1, 48),
        vertical_radius: number("VRADIUS", 4, 1, 32),
        workers: number("WORKERS", default_worker_count() as i64, 1, 32) as usize,
    };
    let generator = TerrainGenerator::new(seed, &registry)?;

    let spawn_x = 0i64;
    let spawn_z = 0i64;
    let spawn_height = generator.height_at(spawn_x, spawn_z) as f64 + 30.0;
    let center = BlockPos::new(spawn_x, 0, spawn_z).chunk();
    let render_origin = ChunkPos::new(center.x, 0, center.z).origin();
    log::info!("world: seed {seed}");

    Ok(WorldView {
        seed,
        render_origin,
        radius_blocks: (config.radius * 32) as f32,
        camera: FlyCamera::new(DVec3::new(spawn_x as f64 + 0.5, spawn_height, spawn_z as f64 + 0.5)),
        streamer: TerrainStreamer::new(generator, &registry, config, render_origin)?,
    })
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.running.is_some() {
            return;
        }
        match self.start(event_loop) {
            Ok(running) => self.running = Some(running),
            Err(error) => self.fail(event_loop, error),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(run) = self.running.as_mut() else {
            return;
        };
        let _ = run.egui_state.on_window_event(&run.window, &event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => run.renderer.resize(size.width, size.height),
            WindowEvent::Focused(false) => {
                self.keys.clear();
                self.set_mouse_captured(false);
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if !self.mouse_captured => self.set_mouse_captured(true),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        repeat,
                        ..
                    },
                ..
            } => match state {
                ElementState::Pressed => {
                    self.keys.insert(code);
                    if !repeat {
                        match code {
                            KeyCode::F3 => self.show_debug = !self.show_debug,
                            KeyCode::Escape => self.set_mouse_captured(false),
                            _ => {}
                        }
                    }
                }
                ElementState::Released => {
                    self.keys.remove(&code);
                }
            },
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.redraw() {
                    self.fail(event_loop, error);
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        // «Сырое» движение мыши: не зависит от положения курсора и границ окна.
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.mouse_captured {
                if let Some(run) = self.running.as_mut() {
                    run.world.camera.rotate(delta.0, delta.1);
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(run) = &self.running {
            run.window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Уничтожаем рендер и окно до выхода из цикла событий.
        self.pending_textures.clear();
        self.running = None;
    }
}

/// Собирает цифры для экрана F3.
fn collect_debug_stats(
    renderer: &Renderer,
    fps: &FpsCounter,
    frame: &FrameStats,
    world: &WorldView,
    stream_ms: f32,
    mouse_captured: bool,
) -> DebugStats {
    let info = renderer.gpu_info();
    let (width, height) = renderer.extent();
    let gpu_kind = match info.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => GpuKind::Discrete,
        vk::PhysicalDeviceType::INTEGRATED_GPU => GpuKind::Integrated,
        vk::PhysicalDeviceType::VIRTUAL_GPU => GpuKind::Virtual,
        vk::PhysicalDeviceType::CPU => GpuKind::Software,
        _ => GpuKind::Other,
    };
    let terrain = renderer.terrain_stats();
    let stream = world.streamer.stats();
    let position = world.camera.position;
    let block = BlockPos::new(
        position.x.floor() as i64,
        position.y.floor() as i64,
        position.z.floor() as i64,
    );
    let chunk = block.chunk();
    DebugStats {
        fps: fps.fps(),
        frame_ms: fps.frame_ms(),
        cpu_ms: frame.cpu_ms,
        wait_ms: frame.wait_ms,
        gpu_ms: frame.gpu_ms,
        gpu_name: info.name.clone(),
        gpu_kind,
        driver: info.driver.clone(),
        vulkan_version: format!(
            "{}.{}.{}",
            vk::api_version_major(info.api_version),
            vk::api_version_minor(info.api_version),
            vk::api_version_patch(info.api_version)
        ),
        width,
        height,
        present_mode: renderer
            .present_mode()
            .map(|mode| format!("{mode:?}"))
            .unwrap_or_else(|| "-".into()),
        validation: renderer.validation_enabled(),
        world: Some(WorldDebug {
            position: [position.x, position.y, position.z],
            chunk: [chunk.x, chunk.y, chunk.z],
            seed: world.seed,
            chunks: terrain.chunks,
            draws: frame.terrain_draws,
            quads: frame.terrain_quads,
            pool_percent: if terrain.pool_capacity_quads > 0 {
                terrain.pool_used_quads as f32 * 100.0 / terrain.pool_capacity_quads as f32
            } else {
                0.0
            },
            loaded: stream.loaded,
            meshed: stream.meshed,
            generating: stream.generating,
            meshing: stream.meshing,
            queued_jobs: stream.queued_jobs,
            pending_uploads: terrain.pending_uploads as usize,
            workers: stream.workers,
            generate_ms: stream.generate_ms,
            mesh_ms: stream.mesh_ms,
            stream_ms,
            upload_kb: frame.upload_bytes as f32 / 1024.0,
        }),
        mouse_captured,
    }
}

//! Экран отладки F3.
//!
//! Показывает цифры, по которым меряется скорость игры: время кадра, позицию, чанки
//! (в памяти / с моделью / в поле зрения / закрыто / нарисовано), квады, очереди задач.

use egui::{Align2, Color32, Frame, Id, Margin, RichText, vec2};

use crate::lang::Lang;

/// Тип видеокарты (для подписи «встроенная/дискретная»).
#[derive(Clone, Copy, Debug)]
pub enum GpuKind {
    Discrete,
    Integrated,
    Virtual,
    Software,
    Other,
}

/// Данные для экрана F3. Собирает модуль приложения каждый кадр.
#[derive(Clone, Debug)]
pub struct DebugStats {
    pub fps: f32,
    pub frame_ms: f32,
    pub cpu_ms: f32,
    /// Ожидание видеокарты и монитора за кадр (при VSync почти весь кадр).
    pub wait_ms: f32,
    pub gpu_ms: Option<f32>,
    pub gpu_name: String,
    pub gpu_kind: GpuKind,
    pub driver: String,
    pub vulkan_version: String,
    pub width: u32,
    pub height: u32,
    pub present_mode: String,
    pub validation: bool,
    /// Сведения о мире (нет, пока мир не загружен).
    pub world: Option<WorldDebug>,
    /// Захвачена ли мышь (если нет — показываем подсказку по управлению).
    pub mouse_captured: bool,
}

/// Сведения о мире для экрана F3.
#[derive(Clone, Debug)]
pub struct WorldDebug {
    pub position: [f64; 3],
    pub chunk: [i64; 3],
    pub seed: u64,
    pub chunks: u32,
    /// Команд отрисовки и квадов нарисовано (по данным видеокарты, с задержкой в пару кадров).
    pub draws: u32,
    pub quads: u64,
    /// Отсечение: чанков в поле зрения, закрытых другими (не нарисованы), нарисовано.
    pub in_frustum: u32,
    pub occluded: u32,
    pub drawn_chunks: u32,
    /// Какие уровни отсечения включены.
    pub frustum_culling: bool,
    pub occlusion_culling: bool,
    pub face_culling: bool,
    pub pool_percent: f32,
    /// Чанков в памяти (сгенерированных).
    pub loaded: usize,
    /// Чанков с моделью на экране.
    pub meshed: usize,
    /// Задачи в работе: генерация и построение моделей.
    pub generating: usize,
    pub meshing: usize,
    /// Задач в очереди рабочих потоков.
    pub queued_jobs: usize,
    /// Моделей, ждущих копирования в видеопамять.
    pub pending_uploads: usize,
    pub workers: usize,
    /// Среднее время одной задачи (мс).
    pub generate_ms: f32,
    pub mesh_ms: f32,
    /// Время работы стримера в главном потоке за кадр (мс).
    pub stream_ms: f32,
    /// Скопировано в видеопамять за кадр (КБ).
    pub upload_kb: f32,
    /// Узлов LOD на экране по уровням (индекс = уровень, 1–4).
    pub lod_shown: [usize; 5],
    /// Узлов LOD строится или ждёт показа.
    pub lod_pending: usize,
    /// Среднее время построения узла LOD (мс).
    pub lod_ms: f32,
    /// Столбцов полных чанков, ждущих показа.
    pub gated_columns: usize,
    /// Дальность обзора (блоков).
    pub view_distance: f32,
    /// Время суток (часы, 0–24) и ускорение хода времени (0 — пауза).
    pub hours: f64,
    pub time_speed: f64,
    /// Склонение пути солнца и луны, градусы.
    pub sun_tilt: f32,
    /// Сетка карты теней: по рёбрам блоков (иначе — квадратная «с солнца»).
    pub shadow_grid_blocks: bool,
    /// Каскадов теней перерисовано в этом кадре и квадов в них.
    pub shadow_cascades: u32,
    pub shadow_quads: u64,
}

/// Рисует экран F3 в левом верхнем углу.
pub fn debug_overlay(ctx: &egui::Context, lang: &Lang, stats: &DebugStats) {
    let kind = match stats.gpu_kind {
        GpuKind::Discrete => lang.tr("debug.gpu.discrete"),
        GpuKind::Integrated => lang.tr("debug.gpu.integrated"),
        GpuKind::Virtual => lang.tr("debug.gpu.virtual"),
        GpuKind::Software => lang.tr("debug.gpu.software"),
        GpuKind::Other => lang.tr("debug.gpu.other"),
    };

    let mut lines = vec![
        lang.fmt(
            "debug.frame",
            &[
                ("fps", &format!("{:.0}", stats.fps)),
                ("ms", &format!("{:.2}", stats.frame_ms)),
            ],
        ),
        lang.fmt(
            "debug.cpu_time",
            &[
                ("ms", &format!("{:.2}", stats.cpu_ms)),
                ("wait", &format!("{:.2}", stats.wait_ms)),
            ],
        ),
        match stats.gpu_ms {
            Some(ms) => lang.fmt("debug.gpu_time", &[("ms", &format!("{ms:.2}"))]),
            None => lang.tr("debug.gpu_time_none").to_string(),
        },
        lang.fmt("debug.gpu", &[("name", &stats.gpu_name), ("kind", &kind)]),
        lang.fmt("debug.driver", &[("driver", &stats.driver)]),
        lang.fmt("debug.vulkan", &[("version", &stats.vulkan_version)]),
        lang.fmt(
            "debug.resolution",
            &[("width", &stats.width), ("height", &stats.height)],
        ),
        lang.fmt("debug.present_mode", &[("mode", &stats.present_mode)]),
    ];
    if stats.validation {
        lines.push(lang.tr("debug.validation_on").to_string());
    }
    if let Some(world) = &stats.world {
        let [x, y, z] = world.position;
        lines.push(lang.fmt(
            "debug.position",
            &[
                ("x", &format!("{x:.1}")),
                ("y", &format!("{y:.1}")),
                ("z", &format!("{z:.1}")),
            ],
        ));
        let [cx, cy, cz] = world.chunk;
        lines.push(lang.fmt("debug.chunk", &[("x", &cx), ("y", &cy), ("z", &cz)]));
        lines.push(lang.fmt("debug.seed", &[("seed", &world.seed)]));
        lines.push(lang.fmt(
            "debug.terrain",
            &[
                ("chunks", &world.chunks),
                ("draws", &world.draws),
                ("quads", &world.quads),
            ],
        ));
        lines.push(lang.fmt(
            "debug.culling",
            &[
                ("frustum", &world.in_frustum),
                ("occluded", &world.occluded),
                ("drawn", &world.drawn_chunks),
            ],
        ));
        let on_off = |on: bool| lang.tr(if on { "common.on" } else { "common.off" }).to_string();
        lines.push(lang.fmt(
            "debug.culling_modes",
            &[
                ("frustum", &on_off(world.frustum_culling)),
                ("occlusion", &on_off(world.occlusion_culling)),
                ("faces", &on_off(world.face_culling)),
            ],
        ));
        lines.push(lang.fmt(
            "debug.pool",
            &[("percent", &format!("{:.1}", world.pool_percent))],
        ));
        lines.push(lang.fmt(
            "debug.stream",
            &[("loaded", &world.loaded), ("meshed", &world.meshed)],
        ));
        lines.push(lang.fmt(
            "debug.jobs",
            &[
                ("generating", &world.generating),
                ("meshing", &world.meshing),
                ("queued", &world.queued_jobs),
                ("uploads", &world.pending_uploads),
                ("workers", &world.workers),
            ],
        ));
        let minutes = (world.hours * 60.0).floor() as u32 % (24 * 60);
        let speed = if world.time_speed == 0.0 {
            lang.tr("debug.time_paused").to_string()
        } else {
            format!("×{}", world.time_speed)
        };
        lines.push(lang.fmt(
            "debug.light",
            &[
                ("time", &format!("{:02}:{:02}", minutes / 60, minutes % 60)),
                ("speed", &speed),
                ("tilt", &format!("{:.0}", world.sun_tilt)),
                (
                    "grid",
                    &lang.tr(if world.shadow_grid_blocks {
                        "debug.shadow_grid.blocks"
                    } else {
                        "debug.shadow_grid.sun"
                    }),
                ),
                ("cascades", &world.shadow_cascades),
                ("quads", &world.shadow_quads),
            ],
        ));
        let [_, l1, l2, l3, l4] = world.lod_shown;
        lines.push(lang.fmt(
            "debug.lod",
            &[
                ("l1", &l1),
                ("l2", &l2),
                ("l3", &l3),
                ("l4", &l4),
                ("pending", &world.lod_pending),
                ("gated", &world.gated_columns),
                ("distance", &format!("{:.0}", world.view_distance)),
                ("ms", &format!("{:.2}", world.lod_ms)),
            ],
        ));
        lines.push(lang.fmt(
            "debug.job_time",
            &[
                ("generate", &format!("{:.2}", world.generate_ms)),
                ("mesh", &format!("{:.2}", world.mesh_ms)),
                ("stream", &format!("{:.2}", world.stream_ms)),
                ("upload", &format!("{:.0}", world.upload_kb)),
            ],
        ));
    }
    if !stats.mouse_captured {
        lines.push(String::new());
        lines.push(lang.tr("hint.capture").to_string());
        lines.push(lang.tr("hint.fly").to_string());
        lines.push(lang.tr("hint.time").to_string());
    }

    egui::Area::new(Id::new("parodia_debug_overlay"))
        .anchor(Align2::LEFT_TOP, vec2(8.0, 8.0))
        .interactable(false)
        .show(ctx, |ui| {
            Frame::new()
                .fill(Color32::from_black_alpha(170))
                .inner_margin(Margin::same(8))
                .corner_radius(4.0)
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(lang.tr("debug.title"))
                            .monospace()
                            .strong()
                            .color(Color32::WHITE),
                    );
                    for line in &lines {
                        ui.label(RichText::new(line).monospace().color(Color32::WHITE));
                    }
                });
        });
}

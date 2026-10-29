//! parodia-render — рендер игры на Vulkan 1.3 (через библиотеку `ash`).
//!
//! Как связаны файлы:
//! - `context.rs`   — подключение к Vulkan: экземпляр, выбор видеокарты, логическое устройство, очередь;
//! - `swapchain.rs` — цепочка картинок, которые по очереди показываются в окне;
//! - `timing.rs`    — замер времени кадра на видеокарте (timestamp-запросы);
//! - `resources.rs` — буферы и картинки в видеопамяти;
//! - `range_alloc.rs` — раздача мест внутри одного большого буфера;
//! - `terrain.rs`   — пул геометрии чанков, таблица чанков и конвейер отрисовки ландшафта;
//! - `culling.rs`   — отсечение невидимого на видеокарте: поле зрения, Hi-Z, стороны граней;
//! - `frame_data.rs` — данные кадра для шейдеров: камера, солнце, луна, небо, туман, каскады;
//! - `shadows.rs`   — каскадные тени: карта теней, положение каскадов, кеш каскадов;
//! - `sky.rs`       — небо;
//! - `renderer.rs`  — главный объект: каждый кадр записывает команды, отправляет их видеокарте и показывает результат.
//!
//! Снаружи модуль используется только через `Renderer`.

mod context;
mod culling;
mod frame_data;
mod range_alloc;
mod renderer;
mod resources;
mod shadows;
mod sky;
mod swapchain;
mod terrain;
mod timing;

pub use ash::vk;
pub use context::GpuInfo;
pub use culling::{CullSettings, CullStats};
pub use renderer::{FrameStats, Renderer, RendererConfig, SceneView, UiFrame};
pub use shadows::ShadowGrid;
pub use terrain::{ChunkMeshId, ChunkUpload, TerrainStats, UPLOAD_BUDGET_BYTES};

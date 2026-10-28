//! parodia-render — рендер игры на Vulkan 1.3 (через библиотеку `ash`).
//!
//! Как связаны файлы:
//! - `context.rs`   — подключение к Vulkan: экземпляр, выбор видеокарты, логическое устройство, очередь;
//! - `swapchain.rs` — цепочка картинок, которые по очереди показываются в окне;
//! - `timing.rs`    — замер времени кадра на видеокарте (timestamp-запросы);
//! - `resources.rs` — буферы и картинки в видеопамяти;
//! - `range_alloc.rs` — раздача мест внутри одного большого буфера;
//! - `terrain.rs`   — пул геометрии чанков, таблица чанков и конвейер отрисовки ландшафта;
//! - `renderer.rs`  — главный объект: каждый кадр записывает команды, отправляет их видеокарте и показывает результат.
//!
//! Снаружи модуль используется только через `Renderer`.

mod context;
mod range_alloc;
mod renderer;
mod resources;
mod swapchain;
mod terrain;
mod timing;

pub use ash::vk;
pub use context::GpuInfo;
pub use renderer::{FrameStats, Renderer, RendererConfig, SceneView, UiFrame};
pub use terrain::{ChunkMeshId, ChunkUpload, TerrainStats, UPLOAD_BUDGET_BYTES};

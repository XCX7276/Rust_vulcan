//! parodia-world — данные мира (без графики).
//!
//! Файлы:
//! - `coords.rs`    — координаты блоков и чанков, пределы мира по высоте;
//! - `block.rs`     — номер блока `BlockId` и реестр свойств блоков из `assets/data/blocks.json`;
//! - `chunk.rs`     — хранение блоков одного чанка (однородный или с палитрой);
//! - `noise.rs`     — детерминированный шум (основа генерации);
//! - `generator.rs` — генератор рельефа: чанк по зерну и координатам.
//!
//! На следующих этапах сюда добавятся хранилище загруженных чанков и правки игрока.

pub mod block;
pub mod chunk;
pub mod coords;
pub mod generator;
pub mod noise;

pub use block::{BlockDef, BlockId, BlockRegistry, RenderKind};
pub use chunk::Chunk;
pub use coords::{BlockPos, ChunkPos, LocalPos};
pub use generator::TerrainGenerator;

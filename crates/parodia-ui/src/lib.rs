//! parodia-ui — интерфейс игры.
//!
//! - `lang.rs`  — переводы: все тексты интерфейса берутся из `assets/lang/ru.json` и `en.json`,
//!   в коде нет русского текста;
//! - `debug_overlay.rs` — экран отладки F3 (FPS, время кадра, видеокарта и т. д.).
//!
//! Экран F3 рисуется через egui — это отладочный инструмент. Игровой интерфейс
//! (меню, инвентарь) позже будет своим, в пиксельном стиле.

mod debug_overlay;
mod lang;

pub use debug_overlay::{DebugStats, GpuKind, WorldDebug, debug_overlay};
pub use lang::Lang;

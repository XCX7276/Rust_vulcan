//! parodia-core — базовый модуль игры.
//!
//! Здесь лежит то, что нужно всем остальным модулям и не зависит от графики:
//! - `env_flags` — чтение флагов окружения `PARODIA_*` (включить/выключить подсистему без пересборки);
//! - `logging`   — журнал игры в файл `%APPDATA%\Parodia\logs\latest.log`;
//! - `frame_timer` — счётчик FPS и времени кадра для экрана отладки F3;
//! - `jobs`      — пул рабочих потоков с очередью по приоритету.

pub mod env_flags;
pub mod frame_timer;
pub mod jobs;
pub mod logging;

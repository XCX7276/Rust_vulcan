//! Точка входа игры Parodia.
//!
//! Что происходит при запуске:
//! 1. включается журнал (`%APPDATA%\Parodia\logs\latest.log`);
//! 2. создаётся цикл событий окна (winit);
//! 3. `App` (файл `app.rs`) создаёт окно, рендер и мир и дальше рисует кадры, пока окно не закроют.
//!
//! Остальные файлы: `camera.rs` — летающая камера, `streaming.rs` — подгрузка и выгрузка мира,
//! `lod.rs` — какие участки рисовать подробно, а какие упрощённо, `perf_log.rs` — строка замера
//! скорости в журнале.
//!
//! В финальной сборке (`--release`) консольное окно скрыто; ошибки смотрите в журнале.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod camera;
mod lod;
mod perf_log;
mod streaming;

use anyhow::Result;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<()> {
    let log_path = parodia_core::logging::init();
    log::info!("Parodia {} starting", env!("CARGO_PKG_VERSION"));
    if let Some(path) = &log_path {
        log::info!("log file: {}", path.display());
    }

    let event_loop = EventLoop::new()?;
    // Poll: рисуем кадры непрерывно, не дожидаясь событий окна.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = app::App::new()?;
    event_loop.run_app(&mut app)?;

    if let Some(error) = app.take_error() {
        log::error!("fatal: {error:#}");
        log::logger().flush();
        return Err(error);
    }
    log::info!("Parodia closed normally");
    log::logger().flush();
    Ok(())
}

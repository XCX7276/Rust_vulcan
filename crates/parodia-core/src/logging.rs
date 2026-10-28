//! Журнал игры (логи).
//!
//! Пишет сообщения сразу в консоль и в файл `%APPDATA%\Parodia\logs\latest.log`.
//! Прошлый запуск сохраняется как `previous.log` — удобно, если игра упала.
//! Уровень подробности: `PARODIA_LOG=trace|debug|info|warn|error` (по умолчанию `info`).
//! Также перехватываются паники (аварийные остановки Rust), чтобы их текст попал в файл.

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use log::LevelFilter;

struct GameLogger {
    file: Option<Mutex<File>>,
    level: LevelFilter,
    start: Instant,
}

impl log::Log for GameLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let seconds = self.start.elapsed().as_secs_f64();
        let line = format!(
            "[{seconds:10.3}] {:<5} {}: {}\n",
            record.level(),
            record.target(),
            record.args()
        );
        eprint!("{line}");
        if let Some(file) = &self.file
            && let Ok(mut file) = file.lock()
        {
            let _ = file.write_all(line.as_bytes());
        }
    }

    fn flush(&self) {
        if let Some(file) = &self.file
            && let Ok(mut file) = file.lock()
        {
            let _ = file.flush();
        }
    }
}

/// Папка данных игры: `%APPDATA%\Parodia` (если APPDATA нет — папка `Parodia` рядом с запуском).
pub fn data_dir() -> PathBuf {
    std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Parodia")
}

/// Включает журнал. Возвращает путь к файлу журнала, если файл удалось создать.
pub fn init() -> Option<PathBuf> {
    let level = match crate::env_flags::text("LOG").as_deref() {
        Some("trace") => LevelFilter::Trace,
        Some("debug") => LevelFilter::Debug,
        Some("warn") => LevelFilter::Warn,
        Some("error") => LevelFilter::Error,
        _ => LevelFilter::Info,
    };

    let dir = data_dir().join("logs");
    let latest = dir.join("latest.log");
    let file = fs::create_dir_all(&dir).ok().and_then(|_| {
        if latest.exists() {
            let previous = dir.join("previous.log");
            let _ = fs::remove_file(&previous);
            let _ = fs::rename(&latest, &previous);
        }
        File::create(&latest).ok()
    });
    let path = file.as_ref().map(|_| latest.clone());

    let logger: &'static GameLogger = Box::leak(Box::new(GameLogger {
        file: file.map(Mutex::new),
        level,
        start: Instant::now(),
    }));
    if log::set_logger(logger).is_ok() {
        log::set_max_level(level);
    }
    install_panic_hook();
    path
}

fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        log::error!(target: "panic", "{info}");
        log::logger().flush();
        default_hook(info);
    }));
}

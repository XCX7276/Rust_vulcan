//! Флаги окружения `PARODIA_<ИМЯ>`.
//!
//! Зачем: любую подсистему (тени, LOD, отсечение и т. д.) можно включить или выключить
//! переменной окружения и сравнить скорость «до/после» на одной и той же сборке.
//! Пример в PowerShell: `$env:PARODIA_VSYNC="0"; cargo run --release`.

/// Читает логический флаг `PARODIA_<name>`.
/// Понимает `1/0`, `true/false`, `on/off`, `yes/no`. Если флага нет или он непонятный — `default`.
pub fn flag(name: &str, default: bool) -> bool {
    match text(name) {
        Some(value) => match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => true,
            "0" | "false" | "off" | "no" => false,
            _ => {
                log::warn!("PARODIA_{name}={value}: unknown value, using default {default}");
                default
            }
        },
        None => default,
    }
}

/// Читает текстовое значение `PARODIA_<name>` (пустая строка считается отсутствием).
pub fn text(name: &str) -> Option<String> {
    std::env::var(format!("PARODIA_{name}"))
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

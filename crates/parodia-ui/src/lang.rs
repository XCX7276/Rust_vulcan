//! Переводы интерфейса.
//!
//! Файлы `assets/lang/*.json` встраиваются прямо в exe при сборке, поэтому игра
//! не зависит от того, лежит ли папка assets рядом.
//! Если в выбранном языке нет ключа — берётся английский, если нет и его — показывается сам ключ
//! (так сразу видно, какой строки не хватает).
//! Подстановки пишутся в тексте как `{name}`.

use std::collections::HashMap;
use std::fmt::Display;

use anyhow::{Context as _, Result};

const RU: &str = include_str!("../../../assets/lang/ru.json");
const EN: &str = include_str!("../../../assets/lang/en.json");

pub struct Lang {
    code: &'static str,
    strings: HashMap<String, String>,
    fallback: HashMap<String, String>,
}

impl Lang {
    /// `code` — "ru" или "en"; любой другой код даёт русский.
    pub fn load(code: &str) -> Result<Self> {
        let fallback: HashMap<String, String> =
            serde_json::from_str(EN).context("assets/lang/en.json is invalid")?;
        let (code, strings) = match code {
            "en" => ("en", fallback.clone()),
            _ => (
                "ru",
                serde_json::from_str(RU).context("assets/lang/ru.json is invalid")?,
            ),
        };
        Ok(Self {
            code,
            strings,
            fallback,
        })
    }

    pub fn code(&self) -> &str {
        self.code
    }

    /// Текст по ключу.
    pub fn tr<'a>(&'a self, key: &'a str) -> &'a str {
        self.strings
            .get(key)
            .or_else(|| self.fallback.get(key))
            .map(String::as_str)
            .unwrap_or(key)
    }

    /// Текст по ключу с подстановками: `fmt("debug.frame", &[("fps", &60)])`.
    pub fn fmt(&self, key: &str, args: &[(&str, &dyn Display)]) -> String {
        let mut text = self.tr(key).to_string();
        for (name, value) in args {
            text = text.replace(&format!("{{{name}}}"), &value.to_string());
        }
        text
    }
}

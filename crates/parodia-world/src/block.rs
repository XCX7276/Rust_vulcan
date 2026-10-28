//! Блоки и их реестр.
//!
//! В мире блок хранится как короткий номер `BlockId` (2 байта), а не строка — так быстро и компактно.
//! Свойства каждого блока (как рисовать, твёрдый ли он) лежат в реестре `BlockRegistry`,
//! который читается из `assets/data/blocks.json`. Номер блока = его место в этом файле,
//! первым всегда идёт воздух (номер 0).
//!
//! Название блока для интерфейса берётся из переводов по ключу `block.<id>`.

use std::collections::HashMap;

use anyhow::{Context as _, Result, bail};
use serde_json::Value;

const BLOCKS_JSON: &str = include_str!("../../../assets/data/blocks.json");

/// Номер блока в реестре.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct BlockId(pub u16);

impl BlockId {
    /// Воздух — всегда номер 0.
    pub const AIR: BlockId = BlockId(0);
}

/// Как блок рисуется. От этого зависит, в какой проход рендера он попадает
/// и закрывает ли он грани соседей.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderKind {
    /// Не рисуется (воздух).
    Invisible,
    /// Непрозрачный: закрывает соседние грани целиком.
    Opaque,
    /// С вырезами (листва, трава-карточки): пиксели либо есть, либо нет.
    Cutout,
    /// Полупрозрачный (вода, стекло): рисуется отдельно, после непрозрачного.
    Transparent,
}

/// Свойства одного блока.
#[derive(Clone, Debug)]
pub struct BlockDef {
    /// Текстовый идентификатор (`stone`, `dirt`…), он же ключ перевода `block.<id>`.
    pub id: String,
    pub render: RenderKind,
    /// Сталкивается ли с ним игрок.
    pub solid: bool,
    /// Цвет блока в sRGB (0..1). TEMPORARY: используется для отрисовки, пока нет текстур;
    /// позже останется только для иконок/карты, если понадобится.
    pub color: [f32; 3],
}

impl BlockDef {
    /// Закрывает ли блок грани соседних блоков (для отсечения невидимых граней).
    pub fn occludes(&self) -> bool {
        self.render == RenderKind::Opaque
    }
}

/// Реестр всех блоков игры.
pub struct BlockRegistry {
    defs: Vec<BlockDef>,
    by_name: HashMap<String, BlockId>,
}

impl BlockRegistry {
    /// Реестр из встроенного файла `assets/data/blocks.json`.
    pub fn builtin() -> Result<Self> {
        Self::from_json(BLOCKS_JSON).context("assets/data/blocks.json")
    }

    /// Разбирает JSON вида `{"blocks": [{"id": "air", "render": "invisible", "solid": false}, ...]}`.
    pub fn from_json(text: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(text)?;
        let list = root
            .get("blocks")
            .and_then(Value::as_array)
            .context("missing \"blocks\" array")?;
        if list.len() > u16::MAX as usize {
            bail!("too many blocks: {}", list.len());
        }

        let mut defs = Vec::with_capacity(list.len());
        let mut by_name = HashMap::with_capacity(list.len());
        for (index, entry) in list.iter().enumerate() {
            let id = entry
                .get("id")
                .and_then(Value::as_str)
                .with_context(|| format!("block #{index}: missing \"id\""))?
                .to_string();
            let render = match entry.get("render").and_then(Value::as_str) {
                Some("invisible") => RenderKind::Invisible,
                Some("opaque") => RenderKind::Opaque,
                Some("cutout") => RenderKind::Cutout,
                Some("transparent") => RenderKind::Transparent,
                other => bail!("block '{id}': bad \"render\" value {other:?}"),
            };
            let solid = entry
                .get("solid")
                .and_then(Value::as_bool)
                .with_context(|| format!("block '{id}': missing \"solid\""))?;
            let color = match entry.get("color") {
                None => [1.0, 0.0, 1.0], // ярко-розовый: сразу видно, что цвет забыли задать
                Some(value) => {
                    let parts: Vec<f32> = value
                        .as_array()
                        .map(|a| a.iter().filter_map(Value::as_f64).map(|v| v as f32).collect())
                        .unwrap_or_default();
                    if parts.len() != 3 {
                        bail!("block '{id}': \"color\" must be [r, g, b]");
                    }
                    [parts[0], parts[1], parts[2]]
                }
            };
            if by_name.insert(id.clone(), BlockId(index as u16)).is_some() {
                bail!("duplicate block id '{id}'");
            }
            defs.push(BlockDef {
                id,
                render,
                solid,
                color,
            });
        }

        match defs.first() {
            Some(air) if air.id == "air" && air.render == RenderKind::Invisible && !air.solid => {}
            _ => bail!("first block must be invisible, non-solid \"air\""),
        }
        Ok(Self { defs, by_name })
    }

    /// Свойства блока по номеру.
    #[inline]
    pub fn get(&self, id: BlockId) -> &BlockDef {
        &self.defs[id.0 as usize]
    }

    /// Номер блока по текстовому идентификатору.
    pub fn id(&self, name: &str) -> Option<BlockId> {
        self.by_name.get(name).copied()
    }

    /// Все блоки по порядку номеров.
    pub fn iter(&self) -> impl Iterator<Item = (BlockId, &BlockDef)> {
        self.defs
            .iter()
            .enumerate()
            .map(|(i, def)| (BlockId(i as u16), def))
    }

    /// Сколько блоков в реестре.
    pub fn len(&self) -> usize {
        self.defs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_registry_loads() {
        let registry = BlockRegistry::builtin().unwrap();
        assert_eq!(registry.id("air"), Some(BlockId::AIR));
        let stone = registry.id("stone").unwrap();
        assert!(registry.get(stone).solid);
        assert!(registry.get(stone).occludes());
        assert!(!registry.get(BlockId::AIR).occludes());
    }

    #[test]
    fn rejects_bad_data() {
        assert!(BlockRegistry::from_json(r#"{"blocks": []}"#).is_err());
        assert!(
            BlockRegistry::from_json(r#"{"blocks": [{"id": "stone", "render": "opaque", "solid": true}]}"#)
                .is_err(),
            "air must be first"
        );
        assert!(
            BlockRegistry::from_json(
                r#"{"blocks": [{"id": "air", "render": "invisible", "solid": false},
                               {"id": "air", "render": "invisible", "solid": false}]}"#
            )
            .is_err(),
            "duplicates"
        );
        assert!(
            BlockRegistry::from_json(
                r#"{"blocks": [{"id": "air", "render": "invisible", "solid": false},
                               {"id": "x", "render": "shiny", "solid": true}]}"#
            )
            .is_err(),
            "unknown render kind"
        );
    }
}

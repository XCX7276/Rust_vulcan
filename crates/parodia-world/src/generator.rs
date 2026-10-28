//! Генератор рельефа.
//!
//! Снаружи генератор устроен окончательно: `generate(ChunkPos) -> Chunk`, детерминированно по зерну,
//! любой чанк в любом порядке и любом потоке. Внутри сейчас простая карта высот из шума.
//! TEMPORARY: на этапе 11 внутренность заменится слоями (континенты, хребты, эрозия, климат, биомы…),
//! интерфейс останется тем же.
//!
//! Слои по вертикали в столбце: камень → 3 блока земли → трава сверху; выше — воздух.
//!
//! **LOD.** Для дальних узлов (`lod_heights`, `generate_lod`) генерируется упрощённая копия:
//! 32×32×32 клеток по 2^уровень блоков, и только около поверхности. Высота берётся в центре
//! клетки: клетка твёрдая, если её центр не выше поверхности. Верхняя твёрдая клетка — трава,
//! следующая — земля, ниже — камень.

use crate::block::{BlockId, BlockRegistry};
use crate::chunk::Chunk;
use crate::coords::{CHUNK_SIZE, ChunkPos, LocalPos};
use crate::noise::fbm_2d;

/// Какие номера блоков использует генератор (берутся из реестра один раз).
#[derive(Clone, Copy, Debug)]
struct Palette {
    stone: BlockId,
    dirt: BlockId,
    grass: BlockId,
}

pub struct TerrainGenerator {
    seed: u64,
    blocks: Palette,
}

/// Средняя высота поверхности.
const BASE_HEIGHT: f64 = 64.0;
/// Размах холмов (± блоков от средней высоты).
const AMPLITUDE: f64 = 48.0;
/// Размер самых крупных форм рельефа в блоках.
const SCALE: f64 = 256.0;
/// Толщина слоя земли над камнем.
const DIRT_DEPTH: i64 = 3;

impl TerrainGenerator {
    pub fn new(seed: u64, registry: &BlockRegistry) -> anyhow::Result<Self> {
        let get = |name: &str| {
            registry
                .id(name)
                .ok_or_else(|| anyhow::anyhow!("generator needs block '{name}'"))
        };
        Ok(Self {
            seed,
            blocks: Palette {
                stone: get("stone")?,
                dirt: get("dirt")?,
                grass: get("grass")?,
            },
        })
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Высота поверхности (Y верхнего твёрдого блока) в столбце (x, z).
    pub fn height_at(&self, x: i64, z: i64) -> i64 {
        let n = fbm_2d(self.seed, x as f64 / SCALE, z as f64 / SCALE, 5, 0.5);
        (BASE_HEIGHT + n * AMPLITUDE).floor() as i64
    }

    /// Наименьшая и наибольшая высота поверхности в столбце чанков (для выбора, какие чанки по Y нужны).
    pub fn height_range(&self, chunk_x: i64, chunk_z: i64) -> (i64, i64) {
        let heights = self.column_heights(chunk_x, chunk_z);
        let min = *heights.iter().min().unwrap_or(&0);
        let max = *heights.iter().max().unwrap_or(&0);
        (min, max)
    }

    /// Высоты всех 32×32 столбцов чанка (индекс = z·32 + x).
    fn column_heights(&self, chunk_x: i64, chunk_z: i64) -> Vec<i64> {
        let base_x = chunk_x * CHUNK_SIZE;
        let base_z = chunk_z * CHUNK_SIZE;
        let mut heights = Vec::with_capacity((CHUNK_SIZE * CHUNK_SIZE) as usize);
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                heights.push(self.height_at(base_x + x, base_z + z));
            }
        }
        heights
    }

    /// LOD: высоты поверхности в центрах 32×32 клеток узла уровня `level` (индекс = z·32 + x).
    /// `node_x, node_z` — координаты узла в единицах его размера (32·2^level блоков).
    pub fn lod_heights(&self, level: u32, node_x: i64, node_z: i64) -> Vec<i64> {
        let cell = 1i64 << level;
        let size = CHUNK_SIZE * cell;
        let (base_x, base_z) = (node_x * size + cell / 2, node_z * size + cell / 2);
        let mut heights = Vec::with_capacity((CHUNK_SIZE * CHUNK_SIZE) as usize);
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                heights.push(self.height_at(base_x + x * cell, base_z + z * cell));
            }
        }
        heights
    }

    /// LOD: то же, что `lod_heights`, но с запасом в одну клетку вокруг узла: 34×34,
    /// индекс = (z + 1)·34 + (x + 1), где x и z от −1 до 32. Крайние клетки — это соседние узлы.
    pub fn lod_heights_padded(&self, level: u32, node_x: i64, node_z: i64) -> Vec<i64> {
        let cell = 1i64 << level;
        let size = CHUNK_SIZE * cell;
        let (base_x, base_z) = (node_x * size + cell / 2, node_z * size + cell / 2);
        let mut heights = Vec::with_capacity(34 * 34);
        for z in -1..=CHUNK_SIZE {
            for x in -1..=CHUNK_SIZE {
                heights.push(self.height_at(base_x + x * cell, base_z + z * cell));
            }
        }
        heights
    }

    /// LOD: упрощённый «чанк» узла уровня `level` на высоте `node_y` (в единицах размера узла).
    /// `heights` — результат `lod_heights` для этого узла.
    pub fn generate_lod(&self, level: u32, node_y: i64, heights: &[i64]) -> Chunk {
        let cell = 1i64 << level;
        let origin_y = node_y * CHUNK_SIZE * cell;
        let top_y = origin_y + CHUNK_SIZE * cell - 1;
        let min = *heights.iter().min().unwrap_or(&0);
        let max = *heights.iter().max().unwrap_or(&0);
        if origin_y + cell / 2 > max {
            return Chunk::filled(BlockId::AIR);
        }
        if top_y + 2 * cell < min {
            return Chunk::filled(self.blocks.stone);
        }
        let blocks = self.blocks;
        Chunk::from_fn(|local: LocalPos| {
            let height = heights[local.z as usize * CHUNK_SIZE as usize + local.x as usize];
            let center = origin_y + local.y as i64 * cell + cell / 2;
            if center > height {
                BlockId::AIR
            } else if center + cell > height {
                blocks.grass
            } else if center + 2 * cell > height {
                blocks.dirt
            } else {
                blocks.stone
            }
        })
    }

    /// Генерирует один чанк.
    pub fn generate(&self, pos: ChunkPos) -> Chunk {
        let origin_y = pos.origin().y;
        let top_y = origin_y + CHUNK_SIZE - 1;
        let heights = self.column_heights(pos.x, pos.z);
        let min = *heights.iter().min().unwrap_or(&0);
        let max = *heights.iter().max().unwrap_or(&0);

        // Быстрые случаи: чанк целиком над поверхностью или целиком в камне.
        if origin_y > max {
            return Chunk::filled(BlockId::AIR);
        }
        if top_y < min - DIRT_DEPTH {
            return Chunk::filled(self.blocks.stone);
        }

        let blocks = self.blocks;
        Chunk::from_fn(|local: LocalPos| {
            let height = heights[local.z as usize * CHUNK_SIZE as usize + local.x as usize];
            let y = origin_y + local.y as i64;
            if y > height {
                BlockId::AIR
            } else if y == height {
                blocks.grass
            } else if y > height - 1 - DIRT_DEPTH {
                blocks.dirt
            } else {
                blocks.stone
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::BlockPos;

    fn generator(seed: u64) -> TerrainGenerator {
        TerrainGenerator::new(seed, &BlockRegistry::builtin().unwrap()).unwrap()
    }

    #[test]
    fn same_seed_same_chunk() {
        let pos = ChunkPos::new(3, 2, -7);
        let a = generator(1337).generate(pos);
        let b = generator(1337).generate(pos);
        for i in 0..crate::coords::CHUNK_VOLUME {
            assert_eq!(a.get_index(i), b.get_index(i));
        }
    }

    #[test]
    fn different_seeds_differ() {
        let a = generator(1).height_at(100, 100);
        let differs = (0..50).any(|i| generator(1).height_at(i * 37, 5) != generator(2).height_at(i * 37, 5));
        assert!(differs, "seeds 1 and 2 gave the same terrain (first height {a})");
    }

    #[test]
    fn chunk_borders_match_column_function() {
        // Блок на границе чанков должен совпадать, из какого бы чанка его ни считали.
        let g = generator(99);
        for &(x, z) in &[(31_i64, 0_i64), (32, 0), (-1, -1), (-32, 64)] {
            let h = g.height_at(x, z);
            let pos = BlockPos::new(x, h, z);
            let chunk = g.generate(pos.chunk());
            assert_eq!(chunk.get(pos.local()), BlockRegistry::builtin().unwrap().id("grass").unwrap());
            let above = pos.offset(0, 1, 0);
            assert_eq!(g.generate(above.chunk()).get(above.local()), BlockId::AIR);
        }
    }

    #[test]
    fn lod_cells_follow_surface() {
        let g = generator(7);
        let registry = BlockRegistry::builtin().unwrap();
        for level in 1..=4u32 {
            let cell = 1i64 << level;
            let size = 32 * cell;
            let (nx, nz) = (-1, 2);
            let heights = g.lod_heights(level, nx, nz);
            assert_eq!(heights, g.lod_heights(level, nx, nz), "deterministic");
            let padded = g.lod_heights_padded(level, nx, nz);
            for z in 0..32 {
                for x in 0..32 {
                    assert_eq!(padded[(z + 1) * 34 + x + 1], heights[z * 32 + x]);
                }
            }
            // Запас справа — первый столбец соседнего узла.
            assert_eq!(padded[34 + 33], g.lod_heights(level, nx + 1, nz)[0]);
            let (lx, lz) = (5usize, 9usize);
            let h = heights[lz * 32 + lx];
            assert_eq!(h, g.height_at(nx * size + lx as i64 * cell + cell / 2, nz * size + lz as i64 * cell + cell / 2));
            // Самая верхняя клетка, чей центр не выше поверхности, — трава; над ней — воздух.
            let cy = (h - cell / 2).div_euclid(cell);
            let node_y = cy.div_euclid(32);
            let chunk = g.generate_lod(level, node_y, &heights);
            let local_y = cy.rem_euclid(32) as u8;
            assert_eq!(chunk.get(LocalPos::new(lx as u8, local_y, lz as u8)), registry.id("grass").unwrap());
            if local_y < 31 {
                assert_eq!(chunk.get(LocalPos::new(lx as u8, local_y + 1, lz as u8)), BlockId::AIR);
            }
        }
    }

    #[test]
    fn deep_and_high_chunks_are_uniform() {
        let g = generator(5);
        assert!(g.generate(ChunkPos::new(0, -50, 0)).is_uniform());
        assert!(g.generate(ChunkPos::new(0, 40, 0)).is_uniform());
    }
}

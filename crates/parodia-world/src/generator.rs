//! Генератор рельефа.
//!
//! Снаружи генератор устроен окончательно: `generate(ChunkPos) -> Chunk`, детерминированно по зерну,
//! любой чанк в любом порядке и любом потоке. Внутри сейчас простая карта высот из шума.
//! TEMPORARY: на этапе 11 внутренность заменится слоями (континенты, хребты, эрозия, климат, биомы…),
//! интерфейс останется тем же.
//!
//! Слои по вертикали в столбце: камень → 3 блока земли → трава сверху; выше — воздух.

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
    fn deep_and_high_chunks_are_uniform() {
        let g = generator(5);
        assert!(g.generate(ChunkPos::new(0, -50, 0)).is_uniform());
        assert!(g.generate(ChunkPos::new(0, 40, 0)).is_uniform());
    }
}

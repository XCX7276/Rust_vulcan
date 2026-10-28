//! Координаты мира.
//!
//! - `BlockPos` — положение блока в мире (целые 64-битные числа: мир не «плывёт» вдали от нуля);
//! - `ChunkPos` — положение чанка (куб 32×32×32 блока);
//! - `LocalPos` — положение блока внутри чанка (0..31 по каждой оси).
//!
//! Перевод блок → чанк делается сдвигом на 5 бит (32 = 2^5): это то же, что деление
//! с округлением вниз, и правильно работает для отрицательных чисел (блок −1 лежит в чанке −1).
//!
//! Мир по высоте ограничен: Y от −3008 до +3007 (188 чанков). По X и Z — бесконечный.

/// Сторона чанка в блоках.
pub const CHUNK_SIZE: i64 = 32;
/// log2(CHUNK_SIZE): сдвиг для перевода блок → чанк.
pub const CHUNK_SHIFT: u32 = 5;
/// Маска для координаты внутри чанка.
pub const CHUNK_MASK: i64 = CHUNK_SIZE - 1;
/// Число блоков в чанке.
pub const CHUNK_VOLUME: usize = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;

/// Нижний блок мира (включительно).
pub const WORLD_MIN_Y: i64 = -3008;
/// Верхний блок мира (включительно).
pub const WORLD_MAX_Y: i64 = 3007;
/// Нижний чанк мира по Y (включительно).
pub const CHUNK_MIN_Y: i64 = WORLD_MIN_Y >> CHUNK_SHIFT;
/// Верхний чанк мира по Y (включительно).
pub const CHUNK_MAX_Y: i64 = WORLD_MAX_Y >> CHUNK_SHIFT;

/// Положение блока в мире.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockPos {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

/// Положение чанка в мире (в чанках, не в блоках).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkPos {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

/// Положение блока внутри чанка: каждая ось 0..31.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocalPos {
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

impl BlockPos {
    pub const fn new(x: i64, y: i64, z: i64) -> Self {
        Self { x, y, z }
    }

    /// В каком чанке лежит блок.
    pub const fn chunk(self) -> ChunkPos {
        ChunkPos {
            x: self.x >> CHUNK_SHIFT,
            y: self.y >> CHUNK_SHIFT,
            z: self.z >> CHUNK_SHIFT,
        }
    }

    /// Где блок лежит внутри своего чанка.
    pub const fn local(self) -> LocalPos {
        LocalPos {
            x: (self.x & CHUNK_MASK) as u8,
            y: (self.y & CHUNK_MASK) as u8,
            z: (self.z & CHUNK_MASK) as u8,
        }
    }

    /// Попадает ли блок в пределы мира по высоте.
    pub const fn in_world(self) -> bool {
        self.y >= WORLD_MIN_Y && self.y <= WORLD_MAX_Y
    }

    /// Блок, сдвинутый на (dx, dy, dz).
    pub const fn offset(self, dx: i64, dy: i64, dz: i64) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
            z: self.z + dz,
        }
    }
}

impl ChunkPos {
    pub const fn new(x: i64, y: i64, z: i64) -> Self {
        Self { x, y, z }
    }

    /// Мировые координаты угла чанка (блок с наименьшими x, y, z).
    pub const fn origin(self) -> BlockPos {
        BlockPos {
            x: self.x << CHUNK_SHIFT,
            y: self.y << CHUNK_SHIFT,
            z: self.z << CHUNK_SHIFT,
        }
    }

    /// Мировые координаты блока по его положению внутри этого чанка.
    pub const fn block(self, local: LocalPos) -> BlockPos {
        let origin = self.origin();
        BlockPos {
            x: origin.x + local.x as i64,
            y: origin.y + local.y as i64,
            z: origin.z + local.z as i64,
        }
    }

    /// Попадает ли чанк в пределы мира по высоте.
    pub const fn in_world(self) -> bool {
        self.y >= CHUNK_MIN_Y && self.y <= CHUNK_MAX_Y
    }
}

impl LocalPos {
    /// Создаёт позицию; координаты должны быть 0..31.
    pub const fn new(x: u8, y: u8, z: u8) -> Self {
        debug_assert!(x < 32 && y < 32 && z < 32);
        Self { x, y, z }
    }

    /// Номер ячейки в массиве чанка. Порядок: X меняется быстрее всего, потом Z, потом Y
    /// (горизонтальные слои подряд — удобно для генерации столбцов и мешинга).
    pub const fn index(self) -> usize {
        (self.y as usize) << 10 | (self.z as usize) << 5 | self.x as usize
    }

    /// Обратное к `index`.
    pub const fn from_index(index: usize) -> Self {
        Self {
            x: (index & 31) as u8,
            z: ((index >> 5) & 31) as u8,
            y: ((index >> 10) & 31) as u8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_limits_are_whole_chunks() {
        assert_eq!(CHUNK_MIN_Y, -94);
        assert_eq!(CHUNK_MAX_Y, 93);
        assert_eq!(ChunkPos::new(0, CHUNK_MIN_Y, 0).origin().y, WORLD_MIN_Y);
        assert_eq!(ChunkPos::new(0, CHUNK_MAX_Y, 0).origin().y + CHUNK_SIZE - 1, WORLD_MAX_Y);
        assert_eq!((CHUNK_MAX_Y - CHUNK_MIN_Y + 1), 188);
    }

    #[test]
    fn negative_coordinates() {
        let pos = BlockPos::new(-1, -1, -33);
        assert_eq!(pos.chunk(), ChunkPos::new(-1, -1, -2));
        assert_eq!(pos.local(), LocalPos::new(31, 31, 31));
        assert_eq!(pos.chunk().block(pos.local()), pos);
    }

    #[test]
    fn far_coordinates_roundtrip() {
        for &(x, z) in &[(1_000_000_i64, -1_000_000_i64), (i64::MAX / 2, i64::MIN / 2)] {
            for y in [WORLD_MIN_Y, -1, 0, 31, 32, WORLD_MAX_Y] {
                let pos = BlockPos::new(x, y, z);
                assert_eq!(pos.chunk().block(pos.local()), pos, "{pos:?}");
            }
        }
    }

    #[test]
    fn local_index_roundtrip() {
        for index in 0..CHUNK_VOLUME {
            assert_eq!(LocalPos::from_index(index).index(), index);
        }
        assert_eq!(LocalPos::new(31, 31, 31).index(), CHUNK_VOLUME - 1);
    }

    #[test]
    fn world_height_check() {
        assert!(BlockPos::new(0, WORLD_MIN_Y, 0).in_world());
        assert!(BlockPos::new(0, WORLD_MAX_Y, 0).in_world());
        assert!(!BlockPos::new(0, WORLD_MIN_Y - 1, 0).in_world());
        assert!(!BlockPos::new(0, WORLD_MAX_Y + 1, 0).in_world());
        assert!(!ChunkPos::new(0, CHUNK_MAX_Y + 1, 0).in_world());
    }
}

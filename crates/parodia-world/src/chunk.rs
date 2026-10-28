//! Хранение блоков чанка (32×32×32 = 32 768 ячеек).
//!
//! Два представления:
//! - `Uniform` — весь чанк из одного блока (воздух над миром, камень глубоко внизу).
//!   Занимает почти ноль памяти. Под землёй и в небе таких чанков подавляющее большинство.
//! - `Paletted` — в чанке есть разные блоки. Хранится короткая «палитра» (список разных блоков
//!   в этом чанке), а в каждой ячейке — не сам блок, а номер в палитре, упакованный в 1, 2, 4, 8
//!   или 16 бит. Чанк с 3 видами блоков занимает 32 768 × 2 бита = 8 КБ вместо 64 КБ.
//!
//! Ширина номера — степень двойки, поэтому номер никогда не пересекает границу 64-битного слова:
//! чтение и запись — один сдвиг и одна маска.
//! Когда разных блоков становится больше, чем помещается, палитра расширяется (больше бит).
//! `compact()` убирает из палитры неиспользуемые блоки и, если остался один, возвращает чанк в `Uniform`.

use crate::block::BlockId;
use crate::coords::{CHUNK_VOLUME, LocalPos};

#[derive(Clone, Debug)]
pub struct Chunk {
    data: ChunkData,
}

#[derive(Clone, Debug)]
enum ChunkData {
    Uniform(BlockId),
    Paletted(Paletted),
}

#[derive(Clone, Debug)]
struct Paletted {
    palette: Vec<BlockId>,
    /// Бит на ячейку: 1, 2, 4, 8 или 16.
    bits: u32,
    words: Vec<u64>,
}

impl Paletted {
    fn new(bits: u32) -> Self {
        let per_word = 64 / bits as usize;
        Self {
            palette: Vec::new(),
            bits,
            words: vec![0; CHUNK_VOLUME.div_ceil(per_word)],
        }
    }

    #[inline]
    fn slot(&self, index: usize) -> (usize, u32, u64) {
        let per_word = 64 / self.bits as usize;
        let word = index / per_word;
        let shift = (index % per_word) as u32 * self.bits;
        let mask = if self.bits == 64 { u64::MAX } else { (1u64 << self.bits) - 1 };
        (word, shift, mask)
    }

    #[inline]
    fn get_index(&self, index: usize) -> usize {
        let (word, shift, mask) = self.slot(index);
        ((self.words[word] >> shift) & mask) as usize
    }

    #[inline]
    fn set_index(&mut self, index: usize, value: usize) {
        let (word, shift, mask) = self.slot(index);
        let w = &mut self.words[word];
        *w = (*w & !(mask << shift)) | ((value as u64 & mask) << shift);
    }

    /// Номер блока в палитре; если блока там нет — добавляет (при необходимости расширяя ширину номера).
    fn palette_index(&mut self, block: BlockId) -> usize {
        if let Some(i) = self.palette.iter().position(|&b| b == block) {
            return i;
        }
        if self.palette.len() >= 1 << self.bits {
            self.grow();
        }
        self.palette.push(block);
        self.palette.len() - 1
    }

    /// Удваивает ширину номера (1 → 2 → 4 → 8 → 16 бит), переписывая все ячейки.
    fn grow(&mut self) {
        let mut wider = Paletted::new(self.bits * 2);
        for index in 0..CHUNK_VOLUME {
            wider.set_index(index, self.get_index(index));
        }
        wider.palette = std::mem::take(&mut self.palette);
        *self = wider;
    }
}

/// Сколько бит нужно на номер для палитры из `len` блоков (минимум 1).
fn bits_for(len: usize) -> u32 {
    match len {
        0..=2 => 1,
        3..=4 => 2,
        5..=16 => 4,
        17..=256 => 8,
        _ => 16,
    }
}

impl Chunk {
    /// Чанк целиком из одного блока.
    pub const fn filled(block: BlockId) -> Self {
        Self {
            data: ChunkData::Uniform(block),
        }
    }

    /// Чанк, где блок каждой ячейки задаёт функция (удобно для генерации).
    pub fn from_fn(mut block_at: impl FnMut(LocalPos) -> BlockId) -> Self {
        let first = block_at(LocalPos::from_index(0));
        let mut chunk = Self::filled(first);
        for index in 1..CHUNK_VOLUME {
            let block = block_at(LocalPos::from_index(index));
            // Ячейки, совпадающие с первым блоком, уже заполнены им (номер 0 в палитре).
            if block != first {
                chunk.set_index(index, block);
            }
        }
        chunk
    }

    /// Блок в ячейке.
    #[inline]
    pub fn get(&self, pos: LocalPos) -> BlockId {
        self.get_index(pos.index())
    }

    #[inline]
    pub fn get_index(&self, index: usize) -> BlockId {
        match &self.data {
            ChunkData::Uniform(block) => *block,
            ChunkData::Paletted(p) => p.palette[p.get_index(index)],
        }
    }

    /// Ставит блок в ячейку.
    #[inline]
    pub fn set(&mut self, pos: LocalPos, block: BlockId) {
        self.set_index(pos.index(), block);
    }

    pub fn set_index(&mut self, index: usize, block: BlockId) {
        if let ChunkData::Uniform(current) = self.data {
            if current == block {
                return;
            }
            let mut p = Paletted::new(1);
            p.palette.push(current); // все ячейки уже имеют номер 0 = прежний блок
            self.data = ChunkData::Paletted(p);
        }
        if let ChunkData::Paletted(p) = &mut self.data {
            let i = p.palette_index(block);
            p.set_index(index, i);
        }
    }

    /// Весь ли чанк из одного блока (без проверки ячеек — только по представлению).
    pub fn is_uniform(&self) -> bool {
        matches!(self.data, ChunkData::Uniform(_))
    }

    /// Если чанк однородный — каким блоком заполнен.
    pub fn uniform_block(&self) -> Option<BlockId> {
        match self.data {
            ChunkData::Uniform(block) => Some(block),
            ChunkData::Paletted(_) => None,
        }
    }

    /// Разные блоки, которые есть в палитре чанка.
    pub fn palette(&self) -> &[BlockId] {
        match &self.data {
            ChunkData::Uniform(block) => std::slice::from_ref(block),
            ChunkData::Paletted(p) => &p.palette,
        }
    }

    /// Убирает из палитры неиспользуемые блоки и сужает номера.
    /// Если остался один блок — чанк становится `Uniform`. Вызывать после серии правок.
    pub fn compact(&mut self) {
        let ChunkData::Paletted(p) = &self.data else {
            return;
        };
        let mut used = vec![false; p.palette.len()];
        for index in 0..CHUNK_VOLUME {
            used[p.get_index(index)] = true;
        }
        let kept: Vec<BlockId> = p
            .palette
            .iter()
            .zip(&used)
            .filter_map(|(&block, &u)| u.then_some(block))
            .collect();
        if kept.len() == 1 {
            self.data = ChunkData::Uniform(kept[0]);
            return;
        }
        if kept.len() == p.palette.len() && bits_for(kept.len()) == p.bits {
            return;
        }
        let mut remap = vec![0usize; p.palette.len()];
        let mut new_index = 0;
        for (old, &u) in used.iter().enumerate() {
            if u {
                remap[old] = new_index;
                new_index += 1;
            }
        }
        let mut packed = Paletted::new(bits_for(kept.len()));
        for index in 0..CHUNK_VOLUME {
            packed.set_index(index, remap[p.get_index(index)]);
        }
        packed.palette = kept;
        self.data = ChunkData::Paletted(packed);
    }

    /// Сколько байт кучи занимает чанк (для оверлея F3 и замеров памяти).
    pub fn heap_bytes(&self) -> usize {
        match &self.data {
            ChunkData::Uniform(_) => 0,
            ChunkData::Paletted(p) => {
                p.words.capacity() * size_of::<u64>()
                    + p.palette.capacity() * size_of::<BlockId>()
            }
        }
    }

    /// Бит на ячейку (0 для однородного чанка).
    pub fn bits_per_block(&self) -> u32 {
        match &self.data {
            ChunkData::Uniform(_) => 0,
            ChunkData::Paletted(p) => p.bits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u16) -> BlockId {
        BlockId(n)
    }

    /// Простой детерминированный генератор чисел для тестов (без сторонних библиотек).
    struct TestRng(u64);
    impl TestRng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn filled_chunk_is_uniform_and_free() {
        let chunk = Chunk::filled(id(3));
        assert!(chunk.is_uniform());
        assert_eq!(chunk.get(LocalPos::new(5, 6, 7)), id(3));
        assert_eq!(chunk.heap_bytes(), 0);
    }

    #[test]
    fn setting_same_block_keeps_uniform() {
        let mut chunk = Chunk::filled(id(1));
        chunk.set(LocalPos::new(0, 0, 0), id(1));
        assert!(chunk.is_uniform());
    }

    #[test]
    fn set_get_roundtrip_with_palette_growth() {
        let mut chunk = Chunk::filled(BlockId::AIR);
        let mut expected = vec![BlockId::AIR; CHUNK_VOLUME];
        let mut rng = TestRng(0x9E37_79B9_7F4A_7C15);
        // 300 разных блоков — палитра пройдёт все ширины 1 → 2 → 4 → 8 → 16 бит.
        for _ in 0..20_000 {
            let index = (rng.next() % CHUNK_VOLUME as u64) as usize;
            let block = id((rng.next() % 300) as u16);
            chunk.set_index(index, block);
            expected[index] = block;
        }
        assert_eq!(chunk.bits_per_block(), 16);
        for (index, &block) in expected.iter().enumerate() {
            assert_eq!(chunk.get_index(index), block, "index {index}");
        }
    }

    #[test]
    fn compact_shrinks_and_returns_to_uniform() {
        let mut chunk = Chunk::filled(id(1));
        for b in 2..10 {
            chunk.set_index(b as usize, id(b));
        }
        assert_eq!(chunk.bits_per_block(), 4);
        // Возвращаем все ячейки к блоку 1 — палитра должна схлопнуться.
        for b in 2..10 {
            chunk.set_index(b as usize, id(1));
        }
        chunk.compact();
        assert!(chunk.is_uniform());
        assert_eq!(chunk.uniform_block(), Some(id(1)));
    }

    #[test]
    fn compact_keeps_data() {
        let mut chunk = Chunk::filled(id(0));
        for b in 1..20u16 {
            chunk.set_index(b as usize * 7, id(b));
        }
        // Удаляем половину разных блоков.
        for b in 1..10u16 {
            chunk.set_index(b as usize * 7, id(0));
        }
        let before: Vec<BlockId> = (0..CHUNK_VOLUME).map(|i| chunk.get_index(i)).collect();
        chunk.compact();
        assert_eq!(chunk.palette().len(), 11);
        assert_eq!(chunk.bits_per_block(), 4);
        let after: Vec<BlockId> = (0..CHUNK_VOLUME).map(|i| chunk.get_index(i)).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn from_fn_matches_function() {
        let chunk = Chunk::from_fn(|p| if p.y < 10 { id(1) } else if p.y < 12 { id(2) } else { BlockId::AIR });
        assert!(!chunk.is_uniform());
        assert_eq!(chunk.get(LocalPos::new(3, 5, 9)), id(1));
        assert_eq!(chunk.get(LocalPos::new(3, 11, 9)), id(2));
        assert_eq!(chunk.get(LocalPos::new(3, 20, 9)), BlockId::AIR);
        let solid = Chunk::from_fn(|_| id(4));
        assert!(solid.is_uniform());
    }

    #[test]
    fn three_block_chunk_uses_two_bits() {
        let chunk = Chunk::from_fn(|p| id((p.y % 3) as u16));
        assert_eq!(chunk.bits_per_block(), 2);
        // 32 768 ячеек × 2 бита = 8 КБ (+ немного на палитру).
        assert!(chunk.heap_bytes() >= CHUNK_VOLUME * 2 / 8);
        assert!(chunk.heap_bytes() < CHUNK_VOLUME * 2 / 8 + 64);
    }
}

//! Распределитель участков внутри одного большого буфера (free-list, «первый подходящий»).
//!
//! Все модели чанков лежат в одном большом буфере видеопамяти. Этот распределитель решает,
//! с какого места положить новый меш, и возвращает место обратно, когда чанк выгружен.
//! Соседние свободные участки при освобождении склеиваются — буфер не «дробится».
//! Единица измерения выбирается снаружи (здесь — квады).

#[derive(Debug)]
pub struct RangeAllocator {
    capacity: u32,
    /// Свободные участки (начало, длина), отсортированы по началу, не соприкасаются.
    free: Vec<(u32, u32)>,
    used: u32,
}

impl RangeAllocator {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            free: vec![(0, capacity)],
            used: 0,
        }
    }

    /// Занимает `len` единиц. Возвращает начало участка или `None`, если места нет.
    pub fn alloc(&mut self, len: u32) -> Option<u32> {
        if len == 0 {
            return None;
        }
        let i = self.free.iter().position(|&(_, l)| l >= len)?;
        let (start, l) = self.free[i];
        if l == len {
            self.free.remove(i);
        } else {
            self.free[i] = (start + len, l - len);
        }
        self.used += len;
        Some(start)
    }

    /// Возвращает участок. Склеивает его с соседними свободными.
    pub fn free(&mut self, start: u32, len: u32) {
        debug_assert!(start + len <= self.capacity);
        self.used -= len;
        let i = self.free.partition_point(|&(s, _)| s < start);
        let joins_prev = i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == start;
        let joins_next = i < self.free.len() && start + len == self.free[i].0;
        match (joins_prev, joins_next) {
            (true, true) => {
                self.free[i - 1].1 += len + self.free[i].1;
                self.free.remove(i);
            }
            (true, false) => self.free[i - 1].1 += len,
            (false, true) => {
                self.free[i].0 = start;
                self.free[i].1 += len;
            }
            (false, false) => self.free.insert(i, (start, len)),
        }
    }

    pub fn used(&self) -> u32 {
        self.used
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_free_merges_back() {
        let mut a = RangeAllocator::new(100);
        let x = a.alloc(10).unwrap();
        let y = a.alloc(20).unwrap();
        let z = a.alloc(30).unwrap();
        assert_eq!((x, y, z), (0, 10, 30));
        assert_eq!(a.used(), 60);
        a.free(y, 20);
        a.free(x, 10);
        a.free(z, 30);
        assert_eq!(a.used(), 0);
        assert_eq!(a.free, vec![(0, 100)]);
    }

    #[test]
    fn reuses_hole_and_reports_full() {
        let mut a = RangeAllocator::new(50);
        let x = a.alloc(20).unwrap();
        let _y = a.alloc(30).unwrap();
        assert!(a.alloc(1).is_none());
        a.free(x, 20);
        assert_eq!(a.alloc(15), Some(0));
        assert_eq!(a.alloc(5), Some(15));
        assert!(a.alloc(1).is_none());
    }
}

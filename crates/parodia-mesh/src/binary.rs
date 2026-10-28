//! Binary greedy meshing.
//!
//! Шаги для одного чанка:
//! 1. Собираем «расширенный» куб 34×34×34: сам чанк в середине (1..=32) и по одному слою
//!    от каждого из 6 соседей по краям — чтобы знать, закрыта ли грань на границе чанка.
//!    Если соседа ещё нет, считаем там воздух (грань рисуется; её уберёт перемешивание,
//!    когда сосед появится — этап 4).
//! 2. Для каждой оси строим «столбцы непрозрачности»: 34 бита в числе u64, бит = непрозрачный блок.
//! 3. Грани в сторону +оси: `столбец & !(столбец >> 1)` (блок есть, а следующий пустой);
//!    в сторону −оси: `столбец & !(столбец << 1)`. Одна операция на 32 блока сразу.
//! 4. Видимые грани раскладываем по «слоям» (срез вдоль оси) и по виду блока в битовые плоскости 32×32.
//! 5. В каждой плоскости жадно сливаем соседние биты в прямоугольники: сначала вдоль строки,
//!    затем вниз, пока следующая строка содержит тот же отрезок.
//!
//! Все рабочие массивы живут в `Mesher` и переиспользуются между чанками — без выделения памяти
//! на каждую грань. Каждому рабочему потоку — свой `Mesher`.

use parodia_world::BlockId;
use parodia_world::chunk::Chunk;
use parodia_world::coords::{CHUNK_VOLUME, LocalPos};

use crate::{Face, pack_quad};

const P: usize = 34; // сторона расширенного куба
const P2: usize = P * P;
const P3: usize = P * P * P;

/// Чанк и его 6 соседей по граням (порядок: +X, −X, +Y, −Y, +Z, −Z). `None` — соседа ещё нет.
pub struct Neighborhood<'a> {
    pub center: &'a Chunk,
    pub neighbors: [Option<&'a Chunk>; 6],
}

/// Одна битовая плоскость 32×32 для одного вида блока.
struct Plane {
    block: BlockId,
    rows: [u32; 32],
}

pub struct Mesher {
    /// Расширенный куб блоков 34³; индекс = (y·34 + z)·34 + x.
    padded: Vec<BlockId>,
    /// Столбцы непрозрачности для осей X, Y, Z; индекс = b·34 + a.
    columns: [Vec<u64>; 3],
    /// Плоскости по (грань, слой): 6 × 32 списков.
    planes: Vec<Vec<Plane>>,
}

impl Default for Mesher {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn pidx(x: usize, y: usize, z: usize) -> usize {
    (y * P + z) * P + x
}

impl Mesher {
    pub fn new() -> Self {
        Self {
            padded: vec![BlockId::AIR; P3],
            columns: [vec![0; P2], vec![0; P2], vec![0; P2]],
            planes: (0..6 * 32).map(|_| Vec::new()).collect(),
        }
    }

    /// Строит квады чанка и дописывает их в `out` (по 2 слова u32 на квад).
    /// `occludes[id]` — закрывает ли блок с номером `id` грани соседей (непрозрачный ли).
    /// Возвращает число квадов.
    pub fn mesh(&mut self, n: &Neighborhood<'_>, occludes: &[bool], out: &mut Vec<u32>) -> usize {
        if !may_have_faces(n, occludes) {
            return 0;
        }
        self.fill_padded(n);
        self.build_columns(occludes);
        self.collect_faces(occludes);
        self.merge_planes(out)
    }

    fn fill_padded(&mut self, n: &Neighborhood<'_>) {
        self.padded.fill(BlockId::AIR);
        let center = n.center;
        match center.uniform_block() {
            Some(block) => {
                for y in 1..=32 {
                    for z in 1..=32 {
                        let row = pidx(1, y, z);
                        self.padded[row..row + 32].fill(block);
                    }
                }
            }
            None => {
                for index in 0..CHUNK_VOLUME {
                    let l = LocalPos::from_index(index);
                    self.padded[pidx(l.x as usize + 1, l.y as usize + 1, l.z as usize + 1)] =
                        center.get_index(index);
                }
            }
        }

        // Пограничные слои соседей. (a, b) пробегают две оси, параллельные грани.
        for (side, neighbor) in n.neighbors.iter().enumerate() {
            let Some(chunk) = neighbor else { continue };
            for a in 0..32u8 {
                for b in 0..32u8 {
                    let (local, padded) = match side {
                        0 => (LocalPos::new(0, a, b), pidx(33, a as usize + 1, b as usize + 1)),
                        1 => (LocalPos::new(31, a, b), pidx(0, a as usize + 1, b as usize + 1)),
                        2 => (LocalPos::new(a, 0, b), pidx(a as usize + 1, 33, b as usize + 1)),
                        3 => (LocalPos::new(a, 31, b), pidx(a as usize + 1, 0, b as usize + 1)),
                        4 => (LocalPos::new(a, b, 0), pidx(a as usize + 1, b as usize + 1, 33)),
                        _ => (LocalPos::new(a, b, 31), pidx(a as usize + 1, b as usize + 1, 0)),
                    };
                    self.padded[padded] = chunk.get(local);
                }
            }
        }
    }

    fn build_columns(&mut self, occludes: &[bool]) {
        for columns in &mut self.columns {
            columns.fill(0);
        }
        for y in 0..P {
            for z in 0..P {
                for x in 0..P {
                    let block = self.padded[pidx(x, y, z)];
                    if occludes[block.0 as usize] {
                        // Ось X: a = y, b = z; ось Y: a = x, b = z; ось Z: a = x, b = y.
                        self.columns[0][z * P + y] |= 1 << x;
                        self.columns[1][z * P + x] |= 1 << y;
                        self.columns[2][y * P + x] |= 1 << z;
                    }
                }
            }
        }
    }

    fn collect_faces(&mut self, occludes: &[bool]) {
        for planes in &mut self.planes {
            planes.clear();
        }
        for axis in 0..3 {
            for b in 1..=32 {
                for a in 1..=32 {
                    let column = self.columns[axis][b * P + a];
                    if column == 0 {
                        continue;
                    }
                    let positive = ((column & !(column >> 1)) >> 1) & 0xFFFF_FFFF;
                    let negative = ((column & !(column << 1)) >> 1) & 0xFFFF_FFFF;
                    for (faces, face_index) in [(positive, axis * 2), (negative, axis * 2 + 1)] {
                        let mut bits = faces;
                        while bits != 0 {
                            let layer = bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            let (x, y, z) = match axis {
                                0 => (layer + 1, a, b),
                                1 => (a, layer + 1, b),
                                _ => (a, b, layer + 1),
                            };
                            let block = self.padded[pidx(x, y, z)];
                            debug_assert!(occludes[block.0 as usize]);
                            let planes = &mut self.planes[face_index * 32 + layer];
                            let plane = match planes.iter_mut().position(|p| p.block == block) {
                                Some(i) => &mut planes[i],
                                None => {
                                    planes.push(Plane {
                                        block,
                                        rows: [0; 32],
                                    });
                                    planes.last_mut().expect("just pushed")
                                }
                            };
                            plane.rows[b - 1] |= 1 << (a - 1);
                        }
                    }
                }
            }
        }
    }

    fn merge_planes(&mut self, out: &mut Vec<u32>) -> usize {
        let mut count = 0;
        for (index, planes) in self.planes.iter_mut().enumerate() {
            let face = Face::ALL[index / 32];
            let layer = (index % 32) as u32;
            for plane in planes.iter_mut() {
                let rows = &mut plane.rows;
                for row in 0..32 {
                    while rows[row] != 0 {
                        let start = rows[row].trailing_zeros();
                        let width = (rows[row] >> start).trailing_ones();
                        let mask = if width == 32 {
                            u32::MAX
                        } else {
                            ((1u32 << width) - 1) << start
                        };
                        rows[row] &= !mask;
                        let mut height = 1;
                        while row + height < 32 && rows[row + height] & mask == mask {
                            rows[row + height] &= !mask;
                            height += 1;
                        }
                        let (u, v) = (start, row as u32);
                        let (x, y, z) = match face {
                            Face::PosX | Face::NegX => (layer, u, v),
                            Face::PosY | Face::NegY => (u, layer, v),
                            Face::PosZ | Face::NegZ => (u, v, layer),
                        };
                        out.extend_from_slice(&pack_quad(
                            x,
                            y,
                            z,
                            width,
                            height as u32,
                            face,
                            plane.block.0,
                        ));
                        count += 1;
                    }
                }
            }
        }
        count
    }
}

/// Быстрая проверка без построения: у пустого чанка граней нет; у сплошного непрозрачного,
/// окружённого со всех сторон сплошными непрозрачными соседями, — тоже.
fn may_have_faces(n: &Neighborhood<'_>, occludes: &[bool]) -> bool {
    match n.center.uniform_block() {
        Some(BlockId::AIR) => false,
        Some(block) if occludes[block.0 as usize] => !n.neighbors.iter().all(|neighbor| {
            neighbor
                .and_then(|c| c.uniform_block())
                .is_some_and(|b| occludes[b.0 as usize])
        }),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unpack_quad;

    const STONE: BlockId = BlockId(1);
    const DIRT: BlockId = BlockId(2);
    const OCCLUDES: [bool; 3] = [false, true, true];

    fn mesh(center: &Chunk, neighbors: [Option<&Chunk>; 6]) -> Vec<[u32; 2]> {
        let mut out = Vec::new();
        let count = Mesher::new().mesh(&Neighborhood { center, neighbors }, &OCCLUDES, &mut out);
        assert_eq!(out.len(), count * 2);
        out.chunks(2).map(|c| [c[0], c[1]]).collect()
    }

    /// Суммарная площадь граней по сторонам — независима от способа слияния.
    fn area_per_face(quads: &[[u32; 2]]) -> [u32; 6] {
        let mut area = [0; 6];
        for q in quads {
            let (_, _, _, w, h, face, _) = unpack_quad(*q);
            area[face as usize] += w * h;
        }
        area
    }

    #[test]
    fn empty_chunk_has_no_quads() {
        assert!(mesh(&Chunk::filled(BlockId::AIR), [None; 6]).is_empty());
    }

    #[test]
    fn single_block_has_six_faces() {
        let mut chunk = Chunk::filled(BlockId::AIR);
        chunk.set(LocalPos::new(5, 6, 7), STONE);
        let quads = mesh(&chunk, [None; 6]);
        assert_eq!(quads.len(), 6);
        for q in &quads {
            let (x, y, z, w, h, _, block) = unpack_quad(*q);
            assert_eq!((x, y, z, w, h, block), (5, 6, 7, 1, 1, 1));
        }
    }

    #[test]
    fn two_blocks_merge() {
        let mut chunk = Chunk::filled(BlockId::AIR);
        chunk.set(LocalPos::new(3, 3, 3), STONE);
        chunk.set(LocalPos::new(4, 3, 3), STONE);
        let quads = mesh(&chunk, [None; 6]);
        assert_eq!(quads.len(), 6, "inner faces hidden, side faces merged");
        assert_eq!(area_per_face(&quads), [1, 1, 2, 2, 2, 2]);
    }

    #[test]
    fn different_blocks_do_not_merge() {
        let mut chunk = Chunk::filled(BlockId::AIR);
        chunk.set(LocalPos::new(3, 3, 3), STONE);
        chunk.set(LocalPos::new(4, 3, 3), DIRT);
        assert_eq!(mesh(&chunk, [None; 6]).len(), 10);
    }

    #[test]
    fn solid_chunk_alone_is_six_big_quads() {
        let quads = mesh(&Chunk::filled(STONE), [None; 6]);
        assert_eq!(quads.len(), 6);
        assert_eq!(area_per_face(&quads), [1024; 6]);
    }

    #[test]
    fn solid_chunk_surrounded_is_empty() {
        let stone = Chunk::filled(STONE);
        assert!(mesh(&stone, [Some(&stone); 6]).is_empty());
    }

    #[test]
    fn neighbor_hides_border_faces() {
        // Пол на y = 0 с камнем под ним (соседний чанк −Y): нижние грани скрыты.
        let floor = Chunk::from_fn(|p| if p.y == 0 { STONE } else { BlockId::AIR });
        let stone = Chunk::filled(STONE);
        let quads = mesh(&floor, [None, None, None, Some(&stone), None, None]);
        let area = area_per_face(&quads);
        assert_eq!(area[Face::NegY as usize], 0);
        assert_eq!(area[Face::PosY as usize], 1024);
        assert_eq!(quads.iter().filter(|q| unpack_quad(**q).5 == Face::PosY as u32).count(), 1);
    }

    #[test]
    fn checkerboard_area_is_exact() {
        // Шахматный порядок: ничего не сливается, у каждого блока все 6 граней видимы.
        let chunk = Chunk::from_fn(|p| if (p.x + p.y + p.z) % 2 == 0 { STONE } else { BlockId::AIR });
        let quads = mesh(&chunk, [None; 6]);
        assert_eq!(quads.len(), 16384 * 6);
    }
}

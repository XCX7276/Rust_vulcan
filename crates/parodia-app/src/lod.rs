//! Дерево LOD: какие участки мира рисовать подробно, а какие — упрощённо.
//!
//! Мир по горизонтали делится на квадраты-«узлы» разных уровней:
//! - уровень 0 — обычный столбец чанков (32 блока), рисуется полными чанками;
//! - уровень L ≥ 1 — квадрат 32·2^L блоков; рисуется упрощённо: 32×32×32 клеток,
//!   каждая клетка — куб 2^L блоков (2, 4, 8, 16 м), и только поверхность.
//!
//! Узел уровня L делится на 4 узла уровня L−1, если камера ближе `split(L)` к его квадрату;
//! `split(L) = k · размер(L)` — «геометрическая» дистанция: чем крупнее узел, тем дальше он
//! дробится. Получаются кольца: около камеры — полные чанки (радиус R), дальше — по кольцу
//! каждого уровня, каждое вдвое шире предыдущего. Листья дерева — то, что рисуется.
//! Листья не перекрываются и без щелей покрывают круг дальности.
//!
//! Чтобы граница уровней не «дрожала», когда камера стоит на ней, узел, который уже был раздроблен,
//! собирается обратно только на 10 % дальше (гистерезис).
//!
//! Переходы без дыр (старый уровень держится, пока не готов новый) делает стример
//! (`streaming.rs`); этот файл только выбирает листья.

use std::collections::HashSet;

use parodia_world::coords::CHUNK_SIZE;

/// Наибольший уровень LOD: клетка 16 м.
pub const MAX_LOD_LEVELS: u32 = 4;
/// Насколько дальше «собирается» обратно уже раздробленный узел.
const HYSTERESIS: f64 = 1.1;

/// Узел дерева: уровень и координаты квадрата в единицах его размера.
/// Для уровня 0 `x, z` — координаты столбца чанков.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeKey {
    pub level: u32,
    pub x: i64,
    pub z: i64,
}

impl NodeKey {
    pub fn new(level: u32, x: i64, z: i64) -> Self {
        Self { level, x, z }
    }

    /// Сторона узла уровня `level` в блоках.
    pub fn size_of(level: u32) -> i64 {
        CHUNK_SIZE << level
    }

    pub fn size(self) -> i64 {
        Self::size_of(self.level)
    }

    /// Предок на уровне `level` (≥ своего).
    pub fn ancestor(self, level: u32) -> NodeKey {
        let shift = level - self.level;
        NodeKey::new(level, self.x >> shift, self.z >> shift)
    }

    /// 4 узла уровнем ниже.
    pub fn children(self) -> [NodeKey; 4] {
        let (x, z, l) = (self.x * 2, self.z * 2, self.level - 1);
        [
            NodeKey::new(l, x, z),
            NodeKey::new(l, x + 1, z),
            NodeKey::new(l, x, z + 1),
            NodeKey::new(l, x + 1, z + 1),
        ]
    }

    /// Лежит ли `other` внутри этого узла (или совпадает с ним).
    pub fn contains(self, other: NodeKey) -> bool {
        other.level <= self.level && other.ancestor(self.level) == self
    }

    /// Расстояние по горизонтали от точки до квадрата узла (0, если точка внутри).
    pub fn distance(self, x: f64, z: f64) -> f64 {
        let size = self.size() as f64;
        let (x0, z0) = (self.x as f64 * size, self.z as f64 * size);
        let dx = (x0 - x).max(0.0).max(x - (x0 + size));
        let dz = (z0 - z).max(0.0).max(z - (z0 + size));
        (dx * dx + dz * dz).sqrt()
    }
}

pub struct LodTree {
    levels: u32,
    /// `split(L) = factor · размер(L)`.
    factor: f64,
    /// Раздробленные узлы и корни прошлого выбора (для гистерезиса).
    refined: HashSet<NodeKey>,
    roots: HashSet<NodeKey>,
}

impl LodTree {
    /// `levels` — сколько уровней LOD (0 — без LOD, только чанки); `radius` — радиус полных
    /// чанков в чанках: узел уровня 1 дробится на чанки ближе `radius · 32` блоков.
    pub fn new(levels: u32, radius: i64) -> Self {
        Self {
            levels: levels.min(MAX_LOD_LEVELS),
            factor: (radius.max(1) as f64) / 2.0,
            refined: HashSet::new(),
            roots: HashSet::new(),
        }
    }

    pub fn levels(&self) -> u32 {
        self.levels
    }

    /// Ближе этого расстояния узел уровня `level` дробится.
    fn split(&self, level: u32) -> f64 {
        self.factor * NodeKey::size_of(level) as f64
    }

    /// Дальность обзора в блоках (граница самого крупного уровня).
    pub fn view_distance(&self) -> f64 {
        self.split(self.levels + 1)
    }

    /// Выбирает листья дерева для камеры в точке (x, z) — что рисовать.
    pub fn select(&mut self, x: f64, z: f64) -> HashSet<NodeKey> {
        let top = self.levels;
        let root_size = NodeKey::size_of(top) as f64;
        let reach = self.view_distance() * HYSTERESIS;
        let (x_min, x_max) = (((x - reach) / root_size).floor() as i64, ((x + reach) / root_size).floor() as i64);
        let (z_min, z_max) = (((z - reach) / root_size).floor() as i64, ((z + reach) / root_size).floor() as i64);

        let mut leaves = HashSet::new();
        let mut refined = HashSet::new();
        let mut roots = HashSet::new();
        for rz in z_min..=z_max {
            for rx in x_min..=x_max {
                let root = NodeKey::new(top, rx, rz);
                let limit = self.view_distance() * self.stretch(&self.roots, root);
                if root.distance(x, z) < limit {
                    roots.insert(root);
                    self.visit(root, x, z, &mut leaves, &mut refined);
                }
            }
        }
        self.refined = refined;
        self.roots = roots;
        leaves
    }

    fn visit(
        &self,
        node: NodeKey,
        x: f64,
        z: f64,
        leaves: &mut HashSet<NodeKey>,
        refined: &mut HashSet<NodeKey>,
    ) {
        let split = self.split(node.level) * self.stretch(&self.refined, node);
        if node.level > 0 && node.distance(x, z) < split {
            refined.insert(node);
            for child in node.children() {
                self.visit(child, x, z, leaves, refined);
            }
        } else {
            leaves.insert(node);
        }
    }

    fn stretch(&self, previous: &HashSet<NodeKey>, node: NodeKey) -> f64 {
        if previous.contains(&node) { HYSTERESIS } else { 1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Каждый столбец чанков в круге дальности покрыт ровно одним листом.
    fn assert_exact_cover(leaves: &HashSet<NodeKey>, x: f64, z: f64, distance: f64) {
        let r = (distance / CHUNK_SIZE as f64) as i64 - 2;
        let (cx, cz) = ((x / 32.0).floor() as i64, (z / 32.0).floor() as i64);
        for dz in -r..=r {
            for dx in -r..=r {
                if dx * dx + dz * dz > r * r {
                    continue;
                }
                let column = NodeKey::new(0, cx + dx, cz + dz);
                let covering = leaves.iter().filter(|leaf| leaf.contains(column)).count();
                assert_eq!(covering, 1, "column {column:?} is covered {covering} times");
            }
        }
    }

    #[test]
    fn ancestors_and_children_are_consistent() {
        let node = NodeKey::new(2, -3, 5);
        for child in node.children() {
            assert_eq!(child.ancestor(2), node);
            assert!(node.contains(child));
        }
        assert_eq!(NodeKey::new(0, -1, -1).ancestor(3), NodeKey::new(3, -1, -1));
        assert_eq!(NodeKey::new(0, 8, 15).ancestor(3), NodeKey::new(3, 1, 1));
        assert!(!NodeKey::new(1, 0, 0).contains(NodeKey::new(0, 2, 0)));
    }

    #[test]
    fn leaves_cover_view_once_without_overlap() {
        let mut tree = LodTree::new(4, 8);
        for &(x, z) in &[(0.5, 0.5), (1000.3, -2000.7), (-123_456.0, 98_765.0)] {
            let leaves = tree.select(x, z);
            assert_exact_cover(&leaves, x, z, tree.view_distance());
        }
    }

    #[test]
    fn full_chunks_around_camera_and_coarse_far_away() {
        let mut tree = LodTree::new(4, 8);
        let leaves = tree.select(10.0, 10.0);
        assert!(leaves.contains(&NodeKey::new(0, 0, 0)));
        // В радиусе R (256 блоков) — только полные чанки.
        assert!(leaves.iter().filter(|l| l.level > 0).all(|l| l.distance(10.0, 10.0) >= 255.0));
        assert!(leaves.iter().any(|l| l.level == 4));
        assert!(tree.view_distance() >= 4000.0);
    }

    #[test]
    fn without_lod_only_chunks() {
        let mut tree = LodTree::new(0, 8);
        let leaves = tree.select(0.0, 0.0);
        assert!(leaves.iter().all(|l| l.level == 0));
        assert_exact_cover(&leaves, 0.0, 0.0, tree.view_distance());
    }

    #[test]
    fn small_moves_do_not_flip_the_border() {
        let mut tree = LodTree::new(4, 8);
        let first = tree.select(0.0, 0.0);
        // Шаг вперёд и обратно: из-за гистерезиса выбор возвращается к прежнему.
        tree.select(20.0, 0.0);
        let back = tree.select(0.0, 0.0);
        let changed = first.symmetric_difference(&back).count();
        assert!(changed <= first.len() / 10, "{changed} leaves changed");
    }
}

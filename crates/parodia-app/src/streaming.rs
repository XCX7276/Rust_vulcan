//! Подгрузка и выгрузка мира вокруг камеры (стример чанков).
//!
//! Главный поток каждый кадр:
//! 1. забирает готовые результаты рабочих потоков (столбцы, сгенерированные чанки, меши);
//! 2. если камера перешла в другой чанк — выгружает то, что стало далеко;
//! 3. выбирает, что делать дальше, сортирует по приоритету и отправляет в рабочие потоки,
//!    не больше `max_in_flight` заданий одновременно (при быстром полёте старые не копятся).
//!
//! Какие чанки нужны (по высоте — два диапазона вместе):
//! - **полоса поверхности** столбца: от самой низкой поверхности среди него и 8 соседних столбцов
//!   (минус запас) до самой высокой (плюс слой воздуха) — поверхность видна всегда, даже с высоты;
//! - **окно вокруг камеры**: ±`vertical_radius` чанков по высоте — под землёй грузится только
//!   то, что рядом, а небо над шахтой не грузится.
//!
//! По горизонтали столбцы полных чанков выбирает дерево LOD (`lod.rs`) — это круг радиуса R.
//! Сведения о столбцах (высоты) нужны на 2 столбца шире, генерация — на 1, модели — ровно там.
//! Модель строится, только когда сгенерированы все 6 соседей чанка, поэтому на стыках и краях
//! нет лишних стенок.
//!
//! **LOD.** Дальше R мир рисуется упрощёнными узлами (2, 4, 8, 16 м). Один узел — одно задание:
//! высоты в центрах клеток → «чанки» узла по высоте около поверхности → модели.
//!
//! **Переходы без дыр.** Старое не убирается, пока новое не готово целиком:
//! - узел дробится (камера приблизилась): его части строятся, а новые столбцы полных чанков
//!   «придерживаются» (`gate`: модели готовы, но не показываются). Когда готово **всё** внутри
//!   узла — части показываются, а узел убирается в том же кадре;
//! - узлы собираются в крупный (камера удалилась): старые столбцы чанков остаются на экране
//!   (`hold`), пока крупный узел не построен, и убираются вместе с его появлением.
//!
//! Приоритет: чем ближе к камере и чем точнее по направлению взгляда, тем раньше.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use glam::{DVec3, Vec3};
use parodia_core::jobs::WorkerPool;
use parodia_mesh::{FaceCounts, Mesher, Neighborhood};
use parodia_render::{ChunkMeshId, ChunkUpload, Renderer};
use parodia_world::coords::{CHUNK_MAX_Y, CHUNK_MIN_Y, CHUNK_SHIFT, CHUNK_SIZE};
use parodia_world::{BlockPos, BlockRegistry, Chunk, ChunkPos, TerrainGenerator};

use crate::lod::{LodTree, NodeKey};

/// Смещения соседей в порядке, который ждёт `Neighborhood`: +X, −X, +Y, −Y, +Z, −Z.
const NEIGHBOR_OFFSETS: [(i64, i64, i64); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Настройки стриминга.
#[derive(Clone, Copy, Debug)]
pub struct StreamConfig {
    /// Радиус отрисовки по горизонтали, в чанках.
    pub radius: i64,
    /// Окно по высоте вокруг камеры, в чанках (в каждую сторону).
    pub vertical_radius: i64,
    /// Число рабочих потоков.
    pub workers: usize,
    /// Уровней LOD (0 — без упрощённого дальнего мира).
    pub lod_levels: u32,
    /// Затенение углов (AO) в моделях полных чанков.
    pub ao: bool,
}

/// Рабочее состояние одного потока: свой построитель мешей со своими буферами.
struct WorkerState {
    mesher: Mesher,
}

/// Результат задания рабочего потока.
enum JobOutput {
    Column {
        x: i64,
        z: i64,
        min: i64,
        max: i64,
    },
    Generated {
        pos: ChunkPos,
        chunk: Chunk,
        micros: u64,
    },
    Meshed {
        pos: ChunkPos,
        version: u32,
        quads: Vec<u32>,
        faces: FaceCounts,
        micros: u64,
    },
    Lod {
        key: NodeKey,
        version: u32,
        meshes: Vec<LodMesh>,
        micros: u64,
    },
}

/// Модель одного «чанка» узла LOD (узел по высоте `node_y`).
struct LodMesh {
    node_y: i64,
    quads: Vec<u32>,
    faces: FaceCounts,
}

enum LodNode {
    /// Строится; номер версии — чтобы отбросить устаревший результат.
    Building(u32),
    /// Модели готовы, ждут показа (см. «переходы без дыр»).
    Built(Vec<LodMesh>),
    /// На экране.
    Shown(Vec<ChunkMeshId>),
}

enum Column {
    Pending,
    /// Наименьшая и наибольшая высота поверхности в столбце.
    Ready(i64, i64),
}

enum Mesh {
    /// Модель ещё не строилась.
    None,
    /// Строится; номер версии — чтобы отбросить устаревший результат.
    Building(u32),
    /// Готова, но придержана: столбец ещё закрыт узлом LOD (см. «переходы без дыр»).
    Held(Vec<u32>, FaceCounts),
    /// Готова (`None` — граней нет или пул переполнен).
    Built(Option<ChunkMeshId>),
}

enum Slot {
    Generating,
    Ready { chunk: Arc<Chunk>, mesh: Mesh },
}

enum Task {
    Column(i64, i64),
    Generate(ChunkPos),
    Mesh(ChunkPos),
    Lod(NodeKey),
}

/// Счётчики для экрана F3.
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamStats {
    pub loaded: usize,
    pub meshed: usize,
    pub generating: usize,
    pub meshing: usize,
    pub queued_jobs: usize,
    pub workers: usize,
    pub generate_ms: f32,
    pub mesh_ms: f32,
    /// Узлов LOD на экране по уровням (индекс = уровень; 0 не используется).
    pub lod_shown: [usize; 5],
    /// Узлов LOD строится или ждёт показа.
    pub lod_pending: usize,
    /// Столбцов полных чанков, ждущих показа (придержаны до готовности соседей по узлу).
    pub gated_columns: usize,
    pub lod_ms: f32,
}

pub struct TerrainStreamer {
    pool: WorkerPool<WorkerState, JobOutput>,
    generator: Arc<TerrainGenerator>,
    occludes: Arc<Vec<bool>>,
    config: StreamConfig,
    render_origin: BlockPos,
    columns: HashMap<(i64, i64), Column>,
    chunks: HashMap<ChunkPos, Slot>,
    tree: LodTree,
    /// Столбец чанков камеры (дерево LOD пересчитывается при его смене).
    camera_column: Option<(i64, i64)>,
    /// Столбцы полных чанков, которые нужны (листья уровня 0).
    mesh_columns: HashSet<(i64, i64)>,
    /// Столбцы, где нужна генерация (на 1 шире) и сведения о высотах (на 2 шире).
    gen_columns: HashSet<(i64, i64)>,
    info_columns: HashSet<(i64, i64)>,
    /// Столбцы полных чанков на экране (нужные и придержанные `hold`).
    shown_columns: HashSet<(i64, i64)>,
    /// Нужные столбцы, которые ещё не показаны: их модели придерживаются.
    gate: HashSet<(i64, i64)>,
    /// Ненужные столбцы, которые остаются на экране, пока их не заменит узел LOD.
    hold: HashSet<(i64, i64)>,
    /// Нужные узлы LOD (листья уровня ≥ 1) и их состояние.
    lod_wanted: HashSet<NodeKey>,
    lod: HashMap<NodeKey, LodNode>,
    /// Что-то изменилось — проверить переходы LOD.
    lod_dirty: bool,
    /// Наборы столбцов изменились — проверить выгрузку.
    sets_changed: bool,
    in_flight: usize,
    max_in_flight: usize,
    next_version: u32,
    center: Option<ChunkPos>,
    needs_scan: bool,
    pool_full_warned: bool,
    generate_time: (u64, u32),
    mesh_time: (u64, u32),
    lod_time: (u64, u32),
}

impl TerrainStreamer {
    pub fn new(
        generator: TerrainGenerator,
        registry: &BlockRegistry,
        config: StreamConfig,
        render_origin: BlockPos,
    ) -> Result<Self> {
        let pool = WorkerPool::new(config.workers, "parodia-worker", || WorkerState {
            mesher: Mesher::new(),
        })?;
        let tree = LodTree::new(config.lod_levels, config.radius);
        log::info!(
            "streaming: radius {} chunks, vertical window ±{} chunks, {} worker threads, {} LOD levels, view distance {:.0} blocks",
            config.radius,
            config.vertical_radius,
            pool.threads(),
            tree.levels(),
            tree.view_distance()
        );
        let occludes: Vec<bool> = registry.iter().map(|(_, def)| def.occludes()).collect();
        Ok(Self {
            max_in_flight: pool.threads() * 4,
            pool,
            generator: Arc::new(generator),
            occludes: Arc::new(occludes),
            config,
            render_origin,
            columns: HashMap::new(),
            chunks: HashMap::new(),
            tree,
            camera_column: None,
            mesh_columns: HashSet::new(),
            gen_columns: HashSet::new(),
            info_columns: HashSet::new(),
            shown_columns: HashSet::new(),
            gate: HashSet::new(),
            hold: HashSet::new(),
            lod_wanted: HashSet::new(),
            lod: HashMap::new(),
            lod_dirty: false,
            sets_changed: false,
            in_flight: 0,
            next_version: 0,
            center: None,
            needs_scan: true,
            pool_full_warned: false,
            generate_time: (0, 0),
            mesh_time: (0, 0),
            lod_time: (0, 0),
        })
    }

    /// Дальность обзора в блоках (вместе с LOD).
    pub fn view_distance(&self) -> f64 {
        self.tree.view_distance()
    }

    /// Вызывать каждый кадр.
    pub fn update(&mut self, renderer: &mut Renderer, camera: DVec3, forward: Vec3) {
        self.receive(renderer);
        let camera_block = BlockPos::new(
            camera.x.floor() as i64,
            camera.y.floor() as i64,
            camera.z.floor() as i64,
        );
        let center = camera_block.chunk();
        if self.camera_column != Some((center.x, center.z)) {
            self.camera_column = Some((center.x, center.z));
            self.reselect(camera);
        }
        if self.center != Some(center) {
            self.center = Some(center);
            self.sets_changed = true;
            self.needs_scan = true;
        }
        if self.lod_dirty {
            self.commit_lod(renderer);
        }
        if self.sets_changed {
            self.sets_changed = false;
            self.unload(renderer, center);
        }
        if self.needs_scan && self.in_flight < self.max_in_flight {
            self.schedule(center, camera, forward);
        }
    }

    pub fn stats(&self) -> StreamStats {
        let mut stats = StreamStats {
            queued_jobs: self.pool.queued(),
            workers: self.pool.threads(),
            generate_ms: average_ms(self.generate_time),
            mesh_ms: average_ms(self.mesh_time),
            lod_ms: average_ms(self.lod_time),
            gated_columns: self.gate.len(),
            ..Default::default()
        };
        for (key, node) in &self.lod {
            match node {
                LodNode::Shown(_) => stats.lod_shown[key.level as usize] += 1,
                _ => stats.lod_pending += 1,
            }
        }
        for slot in self.chunks.values() {
            match slot {
                Slot::Generating => stats.generating += 1,
                Slot::Ready { mesh, .. } => {
                    stats.loaded += 1;
                    match mesh {
                        Mesh::Building(_) => stats.meshing += 1,
                        Mesh::Built(Some(_)) | Mesh::Held(..) => stats.meshed += 1,
                        _ => {}
                    }
                }
            }
        }
        stats
    }

    /// Забирает готовые результаты рабочих потоков.
    fn receive(&mut self, renderer: &mut Renderer) {
        while let Some(output) = self.pool.try_recv() {
            self.in_flight = self.in_flight.saturating_sub(1);
            self.needs_scan = true;
            if !self.gate.is_empty() {
                // Готовность придержанных столбцов могла измениться.
                self.lod_dirty = true;
            }
            match output {
                JobOutput::Column { x, z, min, max } => {
                    if let Some(column) = self.columns.get_mut(&(x, z)) {
                        *column = Column::Ready(min, max);
                    }
                }
                JobOutput::Generated { pos, chunk, micros } => {
                    add_time(&mut self.generate_time, micros);
                    // Если чанк успели выгрузить, пока он генерировался, результат просто отбрасываем.
                    if let Some(slot) = self.chunks.get_mut(&pos)
                        && matches!(slot, Slot::Generating)
                    {
                        *slot = Slot::Ready {
                            chunk: Arc::new(chunk),
                            mesh: Mesh::None,
                        };
                    }
                }
                JobOutput::Meshed {
                    pos,
                    version,
                    quads,
                    faces,
                    micros,
                } => {
                    add_time(&mut self.mesh_time, micros);
                    let origin = self.relative(pos.origin());
                    let Some(Slot::Ready { mesh, .. }) = self.chunks.get_mut(&pos) else {
                        continue;
                    };
                    if !matches!(mesh, Mesh::Building(v) if *v == version) {
                        continue; // устаревший результат
                    }
                    let column = (pos.x, pos.z);
                    *mesh = if self.shown_columns.contains(&column) {
                        upload(renderer, origin, quads, faces, 0, &mut self.pool_full_warned)
                    } else if self.gate.contains(&column) {
                        self.lod_dirty = true;
                        Mesh::Held(quads, faces)
                    } else {
                        Mesh::None // столбец больше не нужен
                    };
                }
                JobOutput::Lod {
                    key,
                    version,
                    meshes,
                    micros,
                } => {
                    add_time(&mut self.lod_time, micros);
                    if let Some(node) = self.lod.get_mut(&key)
                        && matches!(node, LodNode::Building(v) if *v == version)
                    {
                        *node = LodNode::Built(meshes);
                        self.lod_dirty = true;
                    }
                }
            }
        }
    }

    /// Пересчитывает дерево LOD: какие столбцы полных чанков и какие узлы LOD нужны.
    fn reselect(&mut self, camera: DVec3) {
        let leaves = self.tree.select(camera.x, camera.z);
        let mut columns = HashSet::new();
        let mut nodes = HashSet::new();
        for leaf in leaves {
            if leaf.level == 0 {
                columns.insert((leaf.x, leaf.z));
            } else {
                nodes.insert(leaf);
            }
        }

        // Новые столбцы: если место закрыто узлом LOD на экране — придерживаем до готовности.
        for &column in &columns {
            if self.hold.remove(&column)
                || self.shown_columns.contains(&column)
                || self.gate.contains(&column)
            {
                continue;
            }
            if self.covered_by_shown_lod(column) {
                self.gate.insert(column);
            } else {
                self.shown_columns.insert(column);
            }
        }
        // Ушедшие столбцы: держим на экране, пока их не заменит узел LOD; вне дальности — убираем.
        let leaving: Vec<(i64, i64)> = self
            .shown_columns
            .iter()
            .filter(|c| !columns.contains(*c) && !self.hold.contains(*c))
            .copied()
            .collect();
        for column in leaving {
            let key = NodeKey::new(0, column.0, column.1);
            let replaced = (1..=self.tree.levels()).any(|l| nodes.contains(&key.ancestor(l)));
            if replaced {
                self.hold.insert(column);
            } else {
                self.shown_columns.remove(&column);
            }
        }
        self.gate.retain(|c| columns.contains(c));
        // Ненужные узлы, которых нет на экране, забываем (их задания отбросятся по версии).
        self.lod
            .retain(|key, node| nodes.contains(key) || matches!(node, LodNode::Shown(_)));

        self.gen_columns = dilate(&columns, 1);
        self.info_columns = dilate(&columns, 2);
        self.mesh_columns = columns;
        self.lod_wanted = nodes;
        self.lod_dirty = true;
        self.sets_changed = true;
        self.needs_scan = true;
    }

    /// Закрыт ли столбец узлом LOD, который сейчас на экране.
    fn covered_by_shown_lod(&self, column: (i64, i64)) -> bool {
        let key = NodeKey::new(0, column.0, column.1);
        (1..=self.tree.levels())
            .any(|l| matches!(self.lod.get(&key.ancestor(l)), Some(LodNode::Shown(_))))
    }

    fn has_shown_ancestor(&self, key: NodeKey) -> bool {
        (key.level + 1..=self.tree.levels())
            .any(|l| matches!(self.lod.get(&key.ancestor(l)), Some(LodNode::Shown(_))))
    }

    /// Переходы между уровнями: показывает готовое и убирает заменённое — только целыми группами,
    /// чтобы на экране не было ни дыр, ни двойной отрисовки.
    fn commit_lod(&mut self, renderer: &mut Renderer) {
        self.lod_dirty = false;
        let levels = self.tree.levels();

        // 1. Готовые нужные узлы без показанного предка: показываем, а всё мелкое внутри убираем
        //    (узлы собрались в крупный, или узел просто появился на краю дальности).
        let built: Vec<NodeKey> = self
            .lod_wanted
            .iter()
            .filter(|k| matches!(self.lod.get(k), Some(LodNode::Built(_))))
            .copied()
            .collect();
        for key in built {
            if self.has_shown_ancestor(key) {
                continue; // это дробление — ждёт готовности всех частей (шаг 2)
            }
            let inside: Vec<NodeKey> = self
                .lod
                .iter()
                .filter(|(k, n)| **k != key && key.contains(**k) && matches!(n, LodNode::Shown(_)))
                .map(|(k, _)| *k)
                .collect();
            for k in inside {
                self.hide_lod(renderer, k);
            }
            let held: Vec<(i64, i64)> = self
                .hold
                .iter()
                .filter(|c| key.contains(NodeKey::new(0, c.0, c.1)))
                .copied()
                .collect();
            for column in held {
                self.hide_column(renderer, column);
            }
            self.show_lod(renderer, key);
        }

        // 2. Показанные, но больше не нужные узлы.
        let stale: Vec<NodeKey> = self
            .lod
            .iter()
            .filter(|(k, n)| matches!(n, LodNode::Shown(_)) && !self.lod_wanted.contains(*k))
            .map(|(k, _)| *k)
            .collect();
        for node in stale {
            if (node.level + 1..=levels).any(|l| self.lod_wanted.contains(&node.ancestor(l))) {
                continue; // заменится крупным узлом (шаг 1), когда тот будет готов
            }
            let parts: Vec<NodeKey> = self
                .lod_wanted
                .iter()
                .filter(|k| node.contains(**k))
                .copied()
                .collect();
            let columns: Vec<(i64, i64)> = self
                .mesh_columns
                .iter()
                .filter(|c| node.contains(NodeKey::new(0, c.0, c.1)))
                .copied()
                .collect();
            let ready = parts.iter().all(|k| {
                matches!(self.lod.get(k), Some(LodNode::Built(_) | LodNode::Shown(_)))
            }) && columns
                .iter()
                .all(|c| !self.gate.contains(c) || self.column_complete(*c));
            if !ready {
                continue;
            }
            // Дробление: все части готовы — показываем их и убираем узел в одном кадре.
            for part in parts {
                if matches!(self.lod.get(&part), Some(LodNode::Built(_))) {
                    self.show_lod(renderer, part);
                }
            }
            for column in columns {
                if self.gate.contains(&column) {
                    self.release_column(renderer, column);
                }
            }
            self.hide_lod(renderer, node);
        }
    }

    /// Загружает модели готового узла LOD на экран.
    fn show_lod(&mut self, renderer: &mut Renderer, key: NodeKey) {
        let Some(LodNode::Built(meshes)) = self.lod.remove(&key) else {
            return;
        };
        let size = key.size();
        let mut ids = Vec::with_capacity(meshes.len());
        for mesh in meshes {
            let origin = self.relative(BlockPos::new(key.x * size, mesh.node_y * size, key.z * size));
            if let Mesh::Built(Some(id)) = upload(
                renderer,
                origin,
                mesh.quads,
                mesh.faces,
                key.level,
                &mut self.pool_full_warned,
            ) {
                ids.push(id);
            }
        }
        self.lod.insert(key, LodNode::Shown(ids));
    }

    fn hide_lod(&mut self, renderer: &mut Renderer, key: NodeKey) {
        if let Some(LodNode::Shown(ids)) = self.lod.remove(&key) {
            for id in ids {
                renderer.remove_chunk(id);
            }
        }
    }

    /// Показывает придержанный столбец: загружает его готовые модели.
    fn release_column(&mut self, renderer: &mut Renderer, column: (i64, i64)) {
        self.gate.remove(&column);
        self.shown_columns.insert(column);
        let render_origin = self.render_origin;
        for (pos, slot) in self.chunks.iter_mut() {
            if (pos.x, pos.z) != column {
                continue;
            }
            if let Slot::Ready { mesh, .. } = slot
                && matches!(mesh, Mesh::Held(..))
                && let Mesh::Held(quads, faces) = std::mem::replace(mesh, Mesh::None)
            {
                let origin = pos.origin();
                let relative = [
                    (origin.x - render_origin.x) as i32,
                    (origin.y - render_origin.y) as i32,
                    (origin.z - render_origin.z) as i32,
                ];
                *mesh = upload(renderer, relative, quads, faces, 0, &mut self.pool_full_warned);
            }
        }
    }

    /// Убирает с экрана придержанный столбец (его заменил узел LOD).
    fn hide_column(&mut self, renderer: &mut Renderer, column: (i64, i64)) {
        self.hold.remove(&column);
        self.shown_columns.remove(&column);
        for (pos, slot) in self.chunks.iter_mut() {
            if (pos.x, pos.z) == column
                && let Slot::Ready { mesh, .. } = slot
            {
                if let Mesh::Built(Some(id)) = mesh {
                    renderer.remove_chunk(*id);
                }
                *mesh = Mesh::None;
            }
        }
        self.sets_changed = true;
    }

    /// Все модели столбца, которые можно построить, построены (для показа придержанного столбца).
    fn column_complete(&self, column: (i64, i64)) -> bool {
        let Some((low, high)) = self.band(column.0, column.1) else {
            return false;
        };
        (low..=high).all(|y| {
            let pos = ChunkPos::new(column.0, y, column.1);
            match self.chunks.get(&pos) {
                Some(Slot::Ready { mesh, .. }) => match mesh {
                    Mesh::Building(_) => false,
                    // Модель, которую нельзя построить (сосед не будет сгенерирован), не ждём.
                    Mesh::None => self.mesh_blocked(pos),
                    Mesh::Held(..) | Mesh::Built(_) => true,
                },
                _ => false,
            }
        })
    }

    /// Модель чанка не построится никогда: какого-то соседа нет и генерировать его не будут.
    /// (Сосед, который просто ещё не готов, — не повод: тогда модель надо дождаться.)
    fn mesh_blocked(&self, pos: ChunkPos) -> bool {
        NEIGHBOR_OFFSETS.iter().any(|&(x, y, z)| {
            let neighbor = ChunkPos::new(pos.x + x, pos.y + y, pos.z + z);
            neighbor.in_world()
                && !self.chunks.contains_key(&neighbor)
                && !self.will_generate(neighbor)
        })
    }

    /// Будет ли чанк сгенерирован при текущих наборах столбцов и положении камеры.
    fn will_generate(&self, pos: ChunkPos) -> bool {
        if !self.gen_columns.contains(&(pos.x, pos.z)) {
            return false;
        }
        match (self.band(pos.x, pos.z), self.center) {
            (Some(band), Some(center)) => self.wanted_layer(band, pos.y, center.y),
            // Высоты столбца ещё не известны — считаем, что будет.
            _ => true,
        }
    }

    /// Угол относительно точки отсчёта рендера.
    fn relative(&self, origin: BlockPos) -> [i32; 3] {
        [
            (origin.x - self.render_origin.x) as i32,
            (origin.y - self.render_origin.y) as i32,
            (origin.z - self.render_origin.z) as i32,
        ]
    }

    /// Выгружает то, что больше не нужно: данные чанков вдали и модели столбцов не на экране.
    fn unload(&mut self, renderer: &mut Renderer, center: ChunkPos) {
        let mut kept: HashSet<(i64, i64)> = self.mesh_columns.clone();
        kept.extend(self.hold.iter().copied());
        let keep_data = dilate(&kept, 2);
        let keep_info = dilate(&kept, 3);
        self.columns.retain(|column, _| keep_info.contains(column));

        let window = self.config.vertical_radius + 2;
        let mut removed = Vec::new();
        let mut unmeshed = Vec::new();
        for (pos, slot) in &self.chunks {
            let column = (pos.x, pos.z);
            let in_window = (pos.y - center.y).abs() <= window;
            let in_band = self
                .band(pos.x, pos.z)
                .is_some_and(|(lo, hi)| pos.y >= lo - 1 && pos.y <= hi + 1);
            let drop = !keep_data.contains(&column) || !(in_window || in_band);
            let visible = self.shown_columns.contains(&column) || self.gate.contains(&column);
            if drop || !visible {
                if let Slot::Ready {
                    mesh: Mesh::Built(Some(id)),
                    ..
                } = slot
                {
                    renderer.remove_chunk(*id);
                }
                if drop {
                    removed.push(*pos);
                } else if matches!(slot, Slot::Ready { mesh, .. } if !matches!(mesh, Mesh::None)) {
                    unmeshed.push(*pos);
                }
            }
        }
        for pos in removed {
            self.chunks.remove(&pos);
        }
        // Данные остаются (нужны соседям), а модели столбца не на экране — убираем.
        for pos in unmeshed {
            if let Some(Slot::Ready { mesh, .. }) = self.chunks.get_mut(&pos) {
                *mesh = Mesh::None;
            }
        }
    }

    /// Полоса поверхности столбца (в чанках по Y), если известны высоты его и 8 соседних столбцов.
    fn band(&self, x: i64, z: i64) -> Option<(i64, i64)> {
        let (mut min, mut max) = (i64::MAX, i64::MIN);
        for dz in -1..=1 {
            for dx in -1..=1 {
                match self.columns.get(&(x + dx, z + dz)) {
                    Some(Column::Ready(lo, hi)) => {
                        min = min.min(*lo);
                        max = max.max(*hi);
                    }
                    _ => return None,
                }
            }
        }
        // Снизу запас в слой камня (его модель не строится), сверху — слой воздуха.
        let bottom = (((min - 8) >> CHUNK_SHIFT) - 1).max(CHUNK_MIN_Y);
        let top = ((max >> CHUNK_SHIFT) + 1).min(CHUNK_MAX_Y);
        Some((bottom, top))
    }

    /// Нужен ли чанк по высоте: в полосе поверхности или в окне вокруг камеры.
    fn wanted_layer(&self, band: (i64, i64), y: i64, camera_y: i64) -> bool {
        let in_band = y >= band.0 && y <= band.1;
        let in_window = (y - camera_y).abs() <= self.config.vertical_radius
            && (CHUNK_MIN_Y..=CHUNK_MAX_Y).contains(&y);
        in_band || in_window
    }

    /// Выбирает следующие задания и отправляет их рабочим потокам.
    fn schedule(&mut self, center: ChunkPos, camera: DVec3, forward: Vec3) {
        let vertical = self.config.vertical_radius;
        let mut tasks: Vec<(f32, Task)> = Vec::new();

        for &(cx, cz) in &self.info_columns {
            if !self.columns.contains_key(&(cx, cz)) {
                // Сведения о столбце нужны раньше всего остального.
                let (dx, dz) = (cx - center.x, cz - center.z);
                tasks.push(((dx * dx + dz * dz) as f32 * 0.25, Task::Column(cx, cz)));
            }
        }
        for &(cx, cz) in &self.gen_columns {
            let Some(band) = self.band(cx, cz) else {
                continue;
            };
            let meshed = self.mesh_columns.contains(&(cx, cz));
            let low = band.0.min(center.y - vertical).max(CHUNK_MIN_Y);
            let high = band.1.max(center.y + vertical).min(CHUNK_MAX_Y);
            for cy in low..=high {
                if !self.wanted_layer(band, cy, center.y) {
                    continue;
                }
                let pos = ChunkPos::new(cx, cy, cz);
                match self.chunks.get(&pos) {
                    None => tasks.push((priority(pos, camera, forward), Task::Generate(pos))),
                    Some(Slot::Ready {
                        mesh: Mesh::None, ..
                    }) if meshed && self.neighbors_ready(pos) => {
                        // Модели — чуть раньше генерации того же места: быстрее появляется картинка.
                        tasks.push((priority(pos, camera, forward) * 0.8, Task::Mesh(pos)));
                    }
                    _ => {}
                }
            }
        }
        for &key in &self.lod_wanted {
            if !self.lod.contains_key(&key) {
                tasks.push((lod_priority(key, camera, forward), Task::Lod(key)));
            }
        }

        let free = self.max_in_flight - self.in_flight;
        tasks.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Если всё, что нужно, отправлено — до новых результатов или движения камеры не пересчитываем.
        self.needs_scan = tasks.len() > free;
        for (priority, task) in tasks.into_iter().take(free) {
            self.submit(priority, task);
        }
    }

    fn neighbors_ready(&self, pos: ChunkPos) -> bool {
        NEIGHBOR_OFFSETS.iter().all(|&(x, y, z)| {
            let neighbor = ChunkPos::new(pos.x + x, pos.y + y, pos.z + z);
            // За пределами мира по высоте соседа нет и не будет — это не мешает строить модель.
            !neighbor.in_world() || matches!(self.chunks.get(&neighbor), Some(Slot::Ready { .. }))
        })
    }

    fn submit(&mut self, priority: f32, task: Task) {
        match task {
            Task::Column(x, z) => {
                self.columns.insert((x, z), Column::Pending);
                let generator = Arc::clone(&self.generator);
                self.pool.submit(priority, move |_| {
                    let (min, max) = generator.height_range(x, z);
                    JobOutput::Column { x, z, min, max }
                });
            }
            Task::Generate(pos) => {
                self.chunks.insert(pos, Slot::Generating);
                let generator = Arc::clone(&self.generator);
                self.pool.submit(priority, move |_| {
                    let started = Instant::now();
                    let chunk = generator.generate(pos);
                    JobOutput::Generated {
                        pos,
                        chunk,
                        micros: started.elapsed().as_micros() as u64,
                    }
                });
            }
            Task::Mesh(pos) => {
                let Some(Slot::Ready { chunk, .. }) = self.chunks.get(&pos) else {
                    return;
                };
                let center = Arc::clone(chunk);
                let neighbors: [Option<Arc<Chunk>>; 6] = NEIGHBOR_OFFSETS.map(|(x, y, z)| {
                    match self.chunks.get(&ChunkPos::new(pos.x + x, pos.y + y, pos.z + z)) {
                        Some(Slot::Ready { chunk, .. }) => Some(Arc::clone(chunk)),
                        _ => None,
                    }
                });
                let version = self.next_version;
                self.next_version = self.next_version.wrapping_add(1);
                if let Some(Slot::Ready { mesh, .. }) = self.chunks.get_mut(&pos) {
                    *mesh = Mesh::Building(version);
                }
                let occludes = Arc::clone(&self.occludes);
                let ao = self.config.ao;
                self.pool.submit(priority, move |state: &mut WorkerState| {
                    let started = Instant::now();
                    let neighborhood = Neighborhood {
                        center: &center,
                        neighbors: neighbors.each_ref().map(|n| n.as_deref()),
                    };
                    let mut quads = Vec::new();
                    let faces = state.mesher.mesh(&neighborhood, &occludes, ao, &mut quads);
                    JobOutput::Meshed {
                        pos,
                        version,
                        quads,
                        faces,
                        micros: started.elapsed().as_micros() as u64,
                    }
                });
            }
            Task::Lod(key) => {
                let version = self.next_version;
                self.next_version = self.next_version.wrapping_add(1);
                self.lod.insert(key, LodNode::Building(version));
                let generator = Arc::clone(&self.generator);
                let occludes = Arc::clone(&self.occludes);
                self.pool.submit(priority, move |state: &mut WorkerState| {
                    let started = Instant::now();
                    let meshes = build_lod_node(&generator, &mut state.mesher, &occludes, key);
                    JobOutput::Lod {
                        key,
                        version,
                        meshes,
                        micros: started.elapsed().as_micros() as u64,
                    }
                });
            }
        }
        self.in_flight += 1;
    }
}

/// Строит узел LOD целиком: высоты в центрах клеток → «чанки» узла около поверхности → модели.
///
/// Соседи по горизонтали — упрощённые копии соседних узлов, опущенные на одну клетку: на краю
/// узла остаётся стенка («юбка») высотой в клетку, которая закрывает щели между уровнями разной
/// подробности, но не тянется до дна узла. Под нижним «чанком» считается камень — у узла нет
/// дна, и из-под земли он не виден коробкой.
fn build_lod_node(
    generator: &TerrainGenerator,
    mesher: &mut Mesher,
    occludes: &[bool],
    key: NodeKey,
) -> Vec<LodMesh> {
    let padded = generator.lod_heights_padded(key.level, key.x, key.z);
    let at = |x: usize, z: usize| padded[z * 34 + x];
    let heights: Vec<i64> = (0..32 * 32).map(|i| at(i % 32 + 1, i / 32 + 1)).collect();
    let cell = 1i64 << key.level;
    // Высоты соседа со стороны `side` (порядок как у `Neighborhood`), опущенные на клетку.
    // Мешер читает у соседа только слой, прилегающий к узлу, поэтому строка повторяется.
    let side_heights = |side: usize| -> Vec<i64> {
        (0..32 * 32)
            .map(|i| {
                let (x, z) = (i % 32, i / 32);
                let h = match side {
                    0 => at(33, z + 1),
                    1 => at(0, z + 1),
                    4 => at(x + 1, 33),
                    _ => at(x + 1, 0),
                };
                h - cell
            })
            .collect()
    };
    let sides: Vec<(usize, Vec<i64>)> = [0, 1, 4, 5].map(|s| (s, side_heights(s))).into();
    let min = heights.iter().copied().min().unwrap_or(0);
    let max = heights.iter().copied().max().unwrap_or(0);
    let size = key.size();
    let bottom = (min - 2 * cell).div_euclid(size);
    let top = max.div_euclid(size);
    let chunks: Vec<(i64, Chunk)> = (bottom..=top)
        .map(|y| (y, generator.generate_lod(key.level, y, &heights)))
        .collect();
    let solid = generator.generate_lod(key.level, 0, &[i64::MAX / 2; 32 * 32]);
    let mut meshes = Vec::with_capacity(chunks.len());
    for (index, (node_y, chunk)) in chunks.iter().enumerate() {
        let above = chunks.get(index + 1).map(|(_, c)| c);
        let below = index.checked_sub(1).map_or(&solid, |i| &chunks[i].1);
        let mut around: [Option<Chunk>; 6] = Default::default();
        for (side, h) in &sides {
            around[*side] = Some(generator.generate_lod(key.level, *node_y, h));
        }
        let neighborhood = Neighborhood {
            center: chunk,
            neighbors: [
                around[0].as_ref(),
                around[1].as_ref(),
                above,
                Some(below),
                around[4].as_ref(),
                around[5].as_ref(),
            ],
        };
        let mut quads = Vec::new();
        let faces = mesher.mesh(&neighborhood, occludes, false, &mut quads);
        if faces.total() > 0 {
            meshes.push(LodMesh {
                node_y: *node_y,
                quads,
                faces,
            });
        }
    }
    meshes
}

/// Загружает модель в рендер (`level` — уровень LOD, 0 — полный чанк).
fn upload(
    renderer: &mut Renderer,
    origin: [i32; 3],
    quads: Vec<u32>,
    faces: FaceCounts,
    level: u32,
    warned: &mut bool,
) -> Mesh {
    match renderer.upload_chunk(origin, quads, faces.0, level) {
        ChunkUpload::Queued(id) => Mesh::Built(Some(id)),
        ChunkUpload::Empty => Mesh::Built(None),
        ChunkUpload::PoolFull => {
            if !*warned {
                log::warn!("terrain geometry pool is full; some chunks are not drawn");
                *warned = true;
            }
            Mesh::Built(None)
        }
    }
}

/// Набор столбцов, расширенный на `by` столбцов во все стороны (квадратом).
fn dilate(columns: &HashSet<(i64, i64)>, by: i64) -> HashSet<(i64, i64)> {
    let mut out = HashSet::with_capacity(columns.len() * 2);
    for &(x, z) in columns {
        for dz in -by..=by {
            for dx in -by..=by {
                out.insert((x + dx, z + dz));
            }
        }
    }
    out
}

/// Приоритет узла LOD: квадрат расстояния до квадрата узла **в размерах узла** и направление взгляда.
/// Один узел закрывает большую площадь и строится быстро, поэтому кольцо каждого уровня получает
/// тот же приоритет, что полные чанки в 4–8 чанках от камеры, — иначе при полёте узлы не успевают
/// строиться за потоком чанков и смена уровня случается прямо перед камерой.
fn lod_priority(key: NodeKey, camera: DVec3, forward: Vec3) -> f32 {
    let size = key.size() as f64;
    let center = DVec3::new((key.x as f64 + 0.5) * size, camera.y, (key.z as f64 + 0.5) * size);
    let dist = key.distance(camera.x, camera.z) / size;
    let facing = ((center - camera).normalize_or_zero().as_vec3()).dot(forward);
    (dist * dist) as f32 * (1.0 - 0.25 * facing - 0.25 * facing.max(0.0)).max(0.25)
}

/// Приоритет чанка: квадрат расстояния до камеры (в чанках), уменьшенный для чанков впереди по взгляду.
fn priority(pos: ChunkPos, camera: DVec3, forward: Vec3) -> f32 {
    let half = CHUNK_SIZE as f64 / 2.0;
    let center = DVec3::new(
        (pos.x * CHUNK_SIZE) as f64 + half,
        (pos.y * CHUNK_SIZE) as f64 + half,
        (pos.z * CHUNK_SIZE) as f64 + half,
    );
    let delta = ((center - camera) / CHUNK_SIZE as f64).as_vec3();
    let dist2 = delta.length_squared();
    let facing = delta.normalize_or_zero().dot(forward);
    // Впереди (facing = 1) — в 2 раза важнее, позади (facing = −1) — в 1,5 раза менее важно.
    dist2 * (1.0 - 0.25 * facing - 0.25 * facing.max(0.0)).max(0.25)
}

fn add_time(total: &mut (u64, u32), micros: u64) {
    // Скользящее окно: после 1000 замеров начинаем заново, чтобы среднее отражало текущую работу.
    if total.1 >= 1000 {
        *total = (0, 0);
    }
    total.0 += micros;
    total.1 += 1;
}

fn average_ms(total: (u64, u32)) -> f32 {
    if total.1 == 0 {
        0.0
    } else {
        total.0 as f32 / total.1 as f32 / 1000.0
    }
}

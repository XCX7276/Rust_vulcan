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
//! По горизонтали — круги: сведения о столбцах (высоты) — радиус R+2, генерация — R+1,
//! модели — R. Модель строится, только когда сгенерированы все 6 соседей чанка, поэтому
//! на стыках и краях нет лишних стенок. Выгрузка — с запасом (R+3), чтобы чанки не
//! «мигали» при движении туда-обратно на границе.
//!
//! Приоритет: чем ближе к камере и чем точнее по направлению взгляда, тем раньше.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use glam::{DVec3, Vec3};
use parodia_core::jobs::WorkerPool;
use parodia_mesh::{FaceCounts, Mesher, Neighborhood};
use parodia_render::{ChunkMeshId, ChunkUpload, Renderer};
use parodia_world::coords::{CHUNK_MAX_Y, CHUNK_MIN_Y, CHUNK_SHIFT, CHUNK_SIZE};
use parodia_world::{BlockPos, BlockRegistry, Chunk, ChunkPos, TerrainGenerator};

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
}

pub struct TerrainStreamer {
    pool: WorkerPool<WorkerState, JobOutput>,
    generator: Arc<TerrainGenerator>,
    occludes: Arc<Vec<bool>>,
    config: StreamConfig,
    render_origin: BlockPos,
    columns: HashMap<(i64, i64), Column>,
    chunks: HashMap<ChunkPos, Slot>,
    in_flight: usize,
    max_in_flight: usize,
    next_version: u32,
    center: Option<ChunkPos>,
    needs_scan: bool,
    pool_full_warned: bool,
    generate_time: (u64, u32),
    mesh_time: (u64, u32),
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
        log::info!(
            "streaming: radius {} chunks, vertical window ±{} chunks, {} worker threads",
            config.radius,
            config.vertical_radius,
            pool.threads()
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
            in_flight: 0,
            next_version: 0,
            center: None,
            needs_scan: true,
            pool_full_warned: false,
            generate_time: (0, 0),
            mesh_time: (0, 0),
        })
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
        if self.center != Some(center) {
            self.center = Some(center);
            self.unload_far(renderer, center);
            self.needs_scan = true;
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
            ..Default::default()
        };
        for slot in self.chunks.values() {
            match slot {
                Slot::Generating => stats.generating += 1,
                Slot::Ready { mesh, .. } => {
                    stats.loaded += 1;
                    match mesh {
                        Mesh::Building(_) => stats.meshing += 1,
                        Mesh::Built(Some(_)) => stats.meshed += 1,
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
                    let Some(Slot::Ready { mesh, .. }) = self.chunks.get_mut(&pos) else {
                        continue;
                    };
                    if !matches!(mesh, Mesh::Building(v) if *v == version) {
                        continue; // устаревший результат
                    }
                    let origin = pos.origin();
                    let relative = [
                        (origin.x - self.render_origin.x) as i32,
                        (origin.y - self.render_origin.y) as i32,
                        (origin.z - self.render_origin.z) as i32,
                    ];
                    *mesh = match renderer.upload_chunk(relative, quads, faces.0) {
                        ChunkUpload::Queued(id) => Mesh::Built(Some(id)),
                        ChunkUpload::Empty => Mesh::Built(None),
                        ChunkUpload::PoolFull => {
                            if !self.pool_full_warned {
                                log::warn!("terrain geometry pool is full; some chunks are not drawn");
                                self.pool_full_warned = true;
                            }
                            Mesh::Built(None)
                        }
                    };
                }
            }
        }
    }

    /// Выгружает чанки и сведения о столбцах, которые стали далеко от камеры.
    fn unload_far(&mut self, renderer: &mut Renderer, center: ChunkPos) {
        let radius = self.config.radius;
        let keep_columns = radius + 4;
        self.columns.retain(|&(x, z), _| {
            let (dx, dz) = (x - center.x, z - center.z);
            dx * dx + dz * dz <= keep_columns * keep_columns
        });

        let keep = radius + 3;
        let window = self.config.vertical_radius + 2;
        let mut removed = Vec::new();
        for (pos, slot) in &self.chunks {
            let (dx, dz) = (pos.x - center.x, pos.z - center.z);
            let far = dx * dx + dz * dz > keep * keep;
            let in_window = (pos.y - center.y).abs() <= window;
            let in_band = self
                .band(pos.x, pos.z)
                .is_some_and(|(lo, hi)| pos.y >= lo - 1 && pos.y <= hi + 1);
            if far || !(in_window || in_band) {
                removed.push(*pos);
                if let Slot::Ready {
                    mesh: Mesh::Built(Some(id)),
                    ..
                } = slot
                {
                    renderer.remove_chunk(*id);
                }
            }
        }
        for pos in removed {
            self.chunks.remove(&pos);
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
        let radius = self.config.radius;
        let vertical = self.config.vertical_radius;
        let mut tasks: Vec<(f32, Task)> = Vec::new();

        for dz in -(radius + 2)..=(radius + 2) {
            for dx in -(radius + 2)..=(radius + 2) {
                let dist2 = dx * dx + dz * dz;
                if dist2 > (radius + 2) * (radius + 2) {
                    continue;
                }
                let (cx, cz) = (center.x + dx, center.z + dz);
                if !self.columns.contains_key(&(cx, cz)) {
                    // Сведения о столбце нужны раньше всего остального.
                    tasks.push((dist2 as f32 * 0.25, Task::Column(cx, cz)));
                    continue;
                }
                if dist2 > (radius + 1) * (radius + 1) {
                    continue;
                }
                let Some(band) = self.band(cx, cz) else {
                    continue;
                };
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
                        }) if dist2 <= radius * radius && self.neighbors_ready(pos) => {
                            // Модели — чуть раньше генерации того же места: быстрее появляется картинка.
                            tasks.push((priority(pos, camera, forward) * 0.8, Task::Mesh(pos)));
                        }
                        _ => {}
                    }
                }
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
                self.pool.submit(priority, move |state: &mut WorkerState| {
                    let started = Instant::now();
                    let neighborhood = Neighborhood {
                        center: &center,
                        neighbors: neighbors.each_ref().map(|n| n.as_deref()),
                    };
                    let mut quads = Vec::new();
                    let faces = state.mesher.mesh(&neighborhood, &occludes, &mut quads);
                    JobOutput::Meshed {
                        pos,
                        version,
                        quads,
                        faces,
                        micros: started.elapsed().as_micros() as u64,
                    }
                });
            }
        }
        self.in_flight += 1;
    }
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

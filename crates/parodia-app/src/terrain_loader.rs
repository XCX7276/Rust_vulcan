//! Загрузка ландшафта вокруг точки появления.
//!
//! TEMPORARY (этап 3): мир грузится один раз при запуске, в главном потоке, кругом радиуса
//! `PARODIA_RADIUS` чанков. На этапе 4 это заменит фоновая подгрузка в рабочих потоках
//! (подгрузка/выгрузка по мере движения, приоритет по направлению взгляда).
//!
//! Шаги: генерация чанков → построение мешей (с учётом соседей) → загрузка в видеопамять.
//! Чанки по высоте берутся только около поверхности: ниже — сплошной камень (граней нет),
//! выше — воздух. Модель строится только у чанков, у которых сгенерированы все 6 соседей,
//! поэтому на стыках и на краю загруженной области нет лишних стенок.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use parodia_mesh::{Mesher, Neighborhood};
use parodia_render::{ChunkUpload, Renderer};
use parodia_world::coords::{CHUNK_MAX_Y, CHUNK_MIN_Y, CHUNK_SHIFT};
use parodia_world::{BlockPos, BlockRegistry, Chunk, ChunkPos, TerrainGenerator};

/// Итоги загрузки (для журнала и экрана F3).
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadReport {
    pub chunks_generated: usize,
    pub chunks_with_mesh: usize,
    pub quads: usize,
    pub generate_ms: f64,
    pub mesh_ms: f64,
    pub upload_ms: f64,
}

/// Смещения соседей в порядке, который ждёт `Neighborhood`: +X, −X, +Y, −Y, +Z, −Z.
const NEIGHBOR_OFFSETS: [(i64, i64, i64); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

pub fn load_around(
    renderer: &mut Renderer,
    registry: &BlockRegistry,
    generator: &TerrainGenerator,
    center: ChunkPos,
    radius: i64,
    render_origin: BlockPos,
) -> Result<LoadReport> {
    let mut report = LoadReport::default();

    // 1. Генерация.
    // Генерируем на одно кольцо чанков шире, чем рисуем, и по высоте берём запас: модель строится
    // только у чанка, все 6 соседей которого уже есть — иначе на стыке появилась бы лишняя стенка.
    let started = Instant::now();
    let in_circle = |dx: i64, dz: i64, r: i64| dx * dx + dz * dz <= r * r;
    let mut heights: HashMap<(i64, i64), (i64, i64)> = HashMap::new();
    for dz in -(radius + 2)..=(radius + 2) {
        for dx in -(radius + 2)..=(radius + 2) {
            let (cx, cz) = (center.x + dx, center.z + dz);
            heights.insert((cx, cz), generator.height_range(cx, cz));
        }
    }
    let mut chunks: HashMap<ChunkPos, Chunk> = HashMap::new();
    for dz in -(radius + 1)..=(radius + 1) {
        for dx in -(radius + 1)..=(radius + 1) {
            if !in_circle(dx, dz, radius + 1) {
                continue;
            }
            let (cx, cz) = (center.x + dx, center.z + dz);
            // Диапазон высот по самому столбцу и его 8 соседям: соседи по бокам будут на тех же высотах.
            let (mut min_height, mut max_height) = (i64::MAX, i64::MIN);
            for nz in -1..=1 {
                for nx in -1..=1 {
                    if let Some(&(lo, hi)) = heights.get(&(cx + nx, cz + nz)) {
                        min_height = min_height.min(lo);
                        max_height = max_height.max(hi);
                    }
                }
            }
            // Снизу — лишний слой камня (его модель не строится), сверху — слой воздуха.
            let bottom = ((min_height - 8) >> CHUNK_SHIFT).saturating_sub(1).max(CHUNK_MIN_Y);
            let top = ((max_height >> CHUNK_SHIFT) + 1).min(CHUNK_MAX_Y);
            for cy in bottom..=top {
                let pos = ChunkPos::new(cx, cy, cz);
                chunks.insert(pos, generator.generate(pos));
            }
        }
    }
    report.chunks_generated = chunks.len();
    report.generate_ms = started.elapsed().as_secs_f64() * 1000.0;

    // 2. Построение мешей.
    let started = Instant::now();
    let occludes: Vec<bool> = registry.iter().map(|(_, def)| def.occludes()).collect();
    let mut mesher = Mesher::new();
    let mut meshes: Vec<([i32; 3], Vec<u32>)> = Vec::new();
    for (pos, chunk) in &chunks {
        if !in_circle(pos.x - center.x, pos.z - center.z, radius) {
            continue;
        }
        let neighbors = NEIGHBOR_OFFSETS
            .map(|(x, y, z)| chunks.get(&ChunkPos::new(pos.x + x, pos.y + y, pos.z + z)));
        // Пока нет хотя бы одного соседа, модель не строим (её построят, когда сосед появится).
        if neighbors.iter().any(Option::is_none) {
            continue;
        }
        let mut quads = Vec::new();
        let count = mesher.mesh(
            &Neighborhood {
                center: chunk,
                neighbors,
            },
            &occludes,
            &mut quads,
        );
        if count > 0 {
            let origin = pos.origin();
            let relative = [
                (origin.x - render_origin.x) as i32,
                (origin.y - render_origin.y) as i32,
                (origin.z - render_origin.z) as i32,
            ];
            report.quads += count;
            meshes.push((relative, quads));
        }
    }
    report.chunks_with_mesh = meshes.len();
    report.mesh_ms = started.elapsed().as_secs_f64() * 1000.0;

    // 3. Загрузка в видеопамять.
    let started = Instant::now();
    let uploads: Vec<ChunkUpload<'_>> = meshes
        .iter()
        .map(|(origin, quads)| ChunkUpload {
            origin: *origin,
            quads,
        })
        .collect();
    let ids = renderer.upload_chunks(&uploads)?;
    let failed = ids.iter().filter(|id| id.is_none()).count();
    if failed > 0 {
        log::warn!("{failed} chunk meshes were not uploaded (geometry pool full)");
    }
    report.upload_ms = started.elapsed().as_secs_f64() * 1000.0;

    log::info!(
        "terrain: {} chunks generated in {:.0} ms, {} meshed ({} quads, {:.1} MB) in {:.0} ms, uploaded in {:.0} ms",
        report.chunks_generated,
        report.generate_ms,
        report.chunks_with_mesh,
        report.quads,
        report.quads as f64 * 8.0 / 1_048_576.0,
        report.mesh_ms,
        report.upload_ms
    );
    Ok(report)
}

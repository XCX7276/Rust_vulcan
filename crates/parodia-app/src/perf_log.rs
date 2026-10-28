//! Строка замера скорости в журнале.
//!
//! Раз в 10 секунд пишет в журнал средние значения:
//! `perf: <FPS> fps, frame <мс> (worst <худший кадр>), cpu <работа мс>, wait <ожидание мс>, gpu <мс>,
//! draws <N>, quads <N>, upload <МБ/с>, <ширина>x<высота>`.
//! Так замеры можно сравнивать по файлу журнала, а не переписывать с экрана.

use std::time::{Duration, Instant};

use parodia_render::FrameStats;

const INTERVAL: Duration = Duration::from_secs(10);

pub struct PerfLog {
    started: Instant,
    frames: u32,
    cpu_ms: f64,
    wait_ms: f64,
    upload_bytes: u64,
    gpu_ms: f64,
    gpu_frames: u32,
    draws: u64,
    quads: u64,
    worst_frame_ms: f32,
    last_frame_at: Option<Instant>,
}

impl PerfLog {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            frames: 0,
            cpu_ms: 0.0,
            wait_ms: 0.0,
            upload_bytes: 0,
            gpu_ms: 0.0,
            gpu_frames: 0,
            draws: 0,
            quads: 0,
            worst_frame_ms: 0.0,
            last_frame_at: None,
        }
    }

    pub fn add(&mut self, frame: &FrameStats, width: u32, height: u32) {
        let now = Instant::now();
        if let Some(previous) = self.last_frame_at {
            let ms = (now - previous).as_secs_f32() * 1000.0;
            self.worst_frame_ms = self.worst_frame_ms.max(ms);
        }
        self.last_frame_at = Some(now);
        self.frames += 1;
        self.cpu_ms += frame.cpu_ms as f64;
        self.wait_ms += frame.wait_ms as f64;
        self.upload_bytes += frame.upload_bytes;
        self.draws += frame.terrain_draws as u64;
        self.quads += frame.terrain_quads;
        if let Some(gpu) = frame.gpu_ms {
            self.gpu_ms += gpu as f64;
            self.gpu_frames += 1;
        }
        let elapsed = self.started.elapsed();
        if elapsed >= INTERVAL && self.frames > 0 {
            let seconds = elapsed.as_secs_f64();
            let frames = self.frames as f64;
            let gpu = if self.gpu_frames > 0 {
                format!("{:.2}", self.gpu_ms / self.gpu_frames as f64)
            } else {
                "-".into()
            };
            log::info!(
                "perf: {:.0} fps, frame {:.2} ms (worst {:.1} ms), cpu {:.2} ms, wait {:.2} ms, gpu {gpu} ms, draws {:.0}, quads {:.0}, upload {:.2} MB/s, {width}x{height}",
                frames / seconds,
                seconds * 1000.0 / frames,
                self.worst_frame_ms,
                self.cpu_ms / frames,
                self.wait_ms / frames,
                self.draws as f64 / frames,
                self.quads as f64 / frames,
                self.upload_bytes as f64 / seconds / 1_048_576.0,
            );
            let last = self.last_frame_at;
            *self = Self::new();
            self.last_frame_at = last;
        }
    }
}

impl Default for PerfLog {
    fn default() -> Self {
        Self::new()
    }
}

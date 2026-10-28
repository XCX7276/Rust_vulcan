//! Счётчик кадров для экрана отладки F3.
//!
//! Считает кадры за окно в полсекунды и выдаёт средний FPS и среднее время кадра.
//! Усреднение нужно, чтобы цифры на экране не прыгали каждый кадр.

use std::time::{Duration, Instant};

const WINDOW: Duration = Duration::from_millis(500);

pub struct FpsCounter {
    window_start: Instant,
    frames: u32,
    fps: f32,
    frame_ms: f32,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            window_start: Instant::now(),
            frames: 0,
            fps: 0.0,
            frame_ms: 0.0,
        }
    }

    /// Вызывать один раз после каждого показанного кадра.
    pub fn tick(&mut self) {
        self.frames += 1;
        let elapsed = self.window_start.elapsed();
        if elapsed >= WINDOW {
            let seconds = elapsed.as_secs_f32();
            self.fps = self.frames as f32 / seconds;
            self.frame_ms = seconds * 1000.0 / self.frames as f32;
            self.frames = 0;
            self.window_start = Instant::now();
        }
    }

    /// Средний FPS за последние полсекунды.
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// Среднее время кадра (мс) за последние полсекунды.
    pub fn frame_ms(&self) -> f32 {
        self.frame_ms
    }
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self::new()
    }
}

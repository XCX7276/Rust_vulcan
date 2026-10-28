//! Замер времени кадра на видеокарте.
//!
//! В начале и в конце кадра видеокарта записывает свой «счётчик времени» (timestamp).
//! Разница двух отметок = сколько видеокарта работала над кадром.
//! Результат читается не сразу, а когда этот кадр точно завершён (после ожидания его fence),
//! поэтому замер не заставляет процессор ждать видеокарту.

use anyhow::Result;
use ash::{Device, vk};

pub struct GpuTimer {
    pool: vk::QueryPool,
    period_ns: f32,
    written: Vec<bool>,
}

impl GpuTimer {
    /// `slots` — сколько кадров одновременно «в полёте» (на каждый по две отметки).
    pub fn new(device: &Device, slots: usize, period_ns: f32) -> Result<Self> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count((slots * 2) as u32);
        // SAFETY: valid device.
        let pool = unsafe { device.create_query_pool(&info, None) }?;
        Ok(Self {
            pool,
            period_ns,
            written: vec![false; slots],
        })
    }

    /// Первая отметка — в самом начале записи команд кадра.
    pub fn begin(&mut self, device: &Device, cmd: vk::CommandBuffer, slot: usize) {
        let first = (slot * 2) as u32;
        // SAFETY: cmd is in recording state, outside any render pass.
        unsafe {
            device.cmd_reset_query_pool(cmd, self.pool, first, 2);
            device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, self.pool, first);
        }
    }

    /// Вторая отметка — в самом конце кадра.
    pub fn end(&mut self, device: &Device, cmd: vk::CommandBuffer, slot: usize) {
        // SAFETY: cmd is in recording state.
        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pool,
                (slot * 2 + 1) as u32,
            );
        }
        self.written[slot] = true;
    }

    /// Время кадра этого слота в миллисекундах. Вызывать после ожидания fence этого слота.
    pub fn read(&self, device: &Device, slot: usize) -> Option<f32> {
        if !self.written[slot] {
            return None;
        }
        let mut data = [0u64; 2];
        // SAFETY: the slot's fence was waited on, so both queries are available.
        let result = unsafe {
            device.get_query_pool_results(
                self.pool,
                (slot * 2) as u32,
                &mut data,
                vk::QueryResultFlags::TYPE_64,
            )
        };
        result
            .ok()
            .map(|()| data[1].wrapping_sub(data[0]) as f32 * self.period_ns / 1_000_000.0)
    }

    pub fn destroy(self, device: &Device) {
        // SAFETY: caller waited for the device to be idle.
        unsafe { device.destroy_query_pool(self.pool, None) };
    }
}

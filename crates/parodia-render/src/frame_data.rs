//! Данные кадра для шейдеров (uniform-буфер).
//!
//! Всё, что одинаково для всего кадра: камера, солнце и луна, цвета неба и рассеянного света,
//! туман, матрицы каскадов теней. Шейдеры ландшафта, теней и неба читают их из одного буфера
//! (set 1 у ландшафта, set 0 у неба), поэтому push constants больше не нужны для камеры.
//!
//! Буферов по одному на кадр «в полёте»: процессор пишет данные кадра N+1, пока видеокарта
//! ещё читает данные кадра N. Здесь же лежит карта теней (binding 1) — её читает шейдер ландшафта.

use anyhow::Result;
use ash::{Device, vk};
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::resources::{GpuBuffer, SharedAllocator};

/// Точная копия блока `Frame` в шейдерах (std140: только mat4 и vec4 — без «дыр»).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameData {
    pub view_proj: [f32; 16],
    pub inv_view_proj: [f32; 16],
    pub cascades: [[f32; 16]; 4],
    pub cascade_split: [f32; 4],
    pub cascade_texel: [f32; 4],
    pub camera_block: [i32; 4],
    /// xyz — дробная часть позиции камеры; w — дальность обзора.
    pub camera_frac: [f32; 4],
    /// xyz — на источник света; w — 1, если тени включены.
    pub light_dir: [f32; 4],
    /// rgb — свет; w — дальность теней.
    pub light_color: [f32; 4],
    /// rgb — рассеянный свет сверху; w — сила AO (0–1).
    pub ambient_sky: [f32; 4],
    pub ambient_ground: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    /// xyz — на солнце; w — «день» (0 ночь … 1 день).
    pub sun_dir: [f32; 4],
    /// xyz — на луну; w — угловой радиус луны (радианы).
    pub moon_dir: [f32; 4],
    /// rgb — свечение у солнца; w — угловой радиус солнца (радианы).
    pub sun_glow: [f32; 4],
    /// x — плотность у базовой высоты, y — убывание с высотой (на блок), z — высота камеры, w — базовая высота.
    pub fog: [f32; 4],
    /// xy — размер картинки в пикселях.
    pub viewport: [f32; 4],
}

pub struct FrameResources {
    buffers: Vec<GpuBuffer>,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

impl FrameResources {
    pub fn new(device: &Device, allocator: &SharedAllocator, frames_in_flight: usize) -> Result<Self> {
        let mut buffers = Vec::with_capacity(frames_in_flight);
        for _ in 0..frames_in_flight {
            buffers.push(GpuBuffer::new(
                device,
                allocator,
                size_of::<FrameData>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                "frame data",
            )?);
        }
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        // SAFETY: valid device; objects are destroyed in `destroy`.
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }?;
        let count = frames_in_flight as u32;
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: count,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: count,
            },
        ];
        // SAFETY: valid device.
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(count)
                    .pool_sizes(&sizes),
                None,
            )
        }?;
        let layouts = vec![set_layout; frames_in_flight];
        // SAFETY: the pool has room for exactly these sets.
        let sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }?;
        for (set, buffer) in sets.iter().zip(&buffers) {
            let info = [vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer)
                .range(vk::WHOLE_SIZE)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&info);
            // SAFETY: the set is not in use yet.
            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }
        Ok(Self {
            buffers,
            set_layout,
            pool,
            sets,
        })
    }

    pub fn set_layout(&self) -> vk::DescriptorSetLayout {
        self.set_layout
    }

    /// Набор дескрипторов кадра «в полёте» `slot`.
    pub fn set(&self, slot: usize) -> vk::DescriptorSet {
        self.sets[slot]
    }

    /// Карта теней для всех наборов. Вызывать, когда видеокарта не использует наборы.
    pub fn set_shadow_map(&self, device: &Device, view: vk::ImageView, sampler: vk::Sampler) {
        let info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let writes: Vec<vk::WriteDescriptorSet> = self
            .sets
            .iter()
            .map(|set| {
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&info)
            })
            .collect();
        // SAFETY: called at init or after the device is idle.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    /// Записывает данные кадра «в полёте» `slot` (после ожидания его fence).
    pub fn write(&mut self, slot: usize, data: &FrameData) -> Result<()> {
        let mapped = self.buffers[slot]
            .mapped()
            .ok_or_else(|| anyhow::anyhow!("frame data buffer is not host-visible"))?;
        let bytes = bytemuck::bytes_of(data);
        mapped[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    pub fn destroy(&mut self, device: &Device, allocator: &SharedAllocator) {
        // SAFETY: caller waited for the device to be idle.
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
        for buffer in &mut self.buffers {
            buffer.destroy(device, allocator);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_std140_block() {
        // 2 mat4 + 4 mat4 + 15 vec4 = 6·64 + 15·16 байт; каждый член выровнен на 16.
        assert_eq!(size_of::<FrameData>(), 6 * 64 + 15 * 16);
    }
}

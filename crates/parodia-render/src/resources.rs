//! Буферы и картинки в памяти видеокарты.
//!
//! Обёртки над `VkBuffer`/`VkImage` вместе с выделенной под них памятью (через gpu-allocator).
//! Память берётся из больших общих блоков, а не отдельным выделением на каждый буфер.
//! Уничтожать — только когда видеокарта гарантированно закончила с ними работать.

use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result};
use ash::{Device, vk};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

pub type SharedAllocator = Arc<Mutex<Allocator>>;

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    allocation: Option<Allocation>,
}

impl GpuBuffer {
    /// `location`: `GpuOnly` — быстрая память видеокарты; `CpuToGpu` — память, в которую пишет процессор.
    pub fn new(
        device: &Device,
        allocator: &SharedAllocator,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self> {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid device and create info.
        let buffer = unsafe { device.create_buffer(&info, None) }?;
        // SAFETY: buffer was just created.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = allocator
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU allocator mutex poisoned"))?
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| format!("out of GPU memory for buffer '{name}' ({size} bytes)"))?;
        // SAFETY: memory satisfies the buffer's requirements and is not bound elsewhere.
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }?;
        Ok(Self {
            buffer,
            size,
            allocation: Some(allocation),
        })
    }

    /// Указатель для записи процессором (только для `CpuToGpu`).
    pub fn mapped(&mut self) -> Option<&mut [u8]> {
        self.allocation.as_mut().and_then(|a| a.mapped_slice_mut())
    }

    pub fn destroy(&mut self, device: &Device, allocator: &SharedAllocator) {
        if let Some(allocation) = self.allocation.take() {
            if let Ok(mut allocator) = allocator.lock() {
                let _ = allocator.free(allocation);
            }
            // SAFETY: caller guarantees the GPU no longer uses the buffer.
            unsafe { device.destroy_buffer(self.buffer, None) };
        }
    }
}

pub struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    allocation: Option<Allocation>,
}

impl GpuImage {
    /// Картинка глубины (для правильного перекрытия: ближнее закрывает дальнее).
    /// `sampled` — шейдеры могут читать её (нужно для пирамиды глубины Hi-Z).
    pub fn depth(
        device: &Device,
        allocator: &SharedAllocator,
        format: vk::Format,
        extent: vk::Extent2D,
        sampled: bool,
    ) -> Result<Self> {
        let mut usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        if sampled {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        Self::new_2d(
            device,
            allocator,
            format,
            extent,
            1,
            usage,
            vk::ImageAspectFlags::DEPTH,
            "depth",
        )
    }

    /// Двумерная картинка с `mip_levels` уровнями; `view` охватывает все уровни.
    #[allow(clippy::too_many_arguments)]
    pub fn new_2d(
        device: &Device,
        allocator: &SharedAllocator,
        format: vk::Format,
        extent: vk::Extent2D,
        mip_levels: u32,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
        name: &str,
    ) -> Result<Self> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: valid device and create info.
        let image = unsafe { device.create_image(&info, None) }?;
        // SAFETY: image was just created.
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = allocator
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU allocator mutex poisoned"))?
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| format!("out of GPU memory for image '{name}'"))?;
        // SAFETY: memory satisfies the image's requirements.
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) }?;
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });
        // SAFETY: image is bound to memory.
        let view = unsafe { device.create_image_view(&view_info, None) }?;
        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
        })
    }

    pub fn destroy(&mut self, device: &Device, allocator: &SharedAllocator) {
        if let Some(allocation) = self.allocation.take() {
            // SAFETY: caller guarantees the GPU no longer uses the image.
            unsafe {
                device.destroy_image_view(self.view, None);
                device.destroy_image(self.image, None);
            }
            if let Ok(mut allocator) = allocator.lock() {
                let _ = allocator.free(allocation);
            }
        }
    }
}

pub fn depth_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::DEPTH,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

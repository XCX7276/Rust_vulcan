//! Цепочка показа кадров (swapchain).
//!
//! Окно показывает одну картинку, пока видеокарта рисует следующую в другую.
//! Swapchain — набор таких картинок (обычно 3). Его приходится пересоздавать,
//! когда меняется размер окна.
//!
//! Режим показа:
//! - VSync включён → FIFO (кадры идут с частотой монитора, без разрывов);
//! - VSync выключен → MAILBOX, если есть, иначе IMMEDIATE (максимум FPS, нужно для замеров).

use anyhow::Result;
use ash::{Device, vk};

use crate::context::GpuContext;

pub struct Swapchain {
    loader: ash::khr::swapchain::Device,
    pub handle: vk::SwapchainKHR,
    pub format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    /// Сигнал «картинка дорисована, можно показывать» — по одному на картинку swapchain.
    pub render_finished: Vec<vk::Semaphore>,
}

impl Swapchain {
    /// Создаёт swapchain. Возвращает `None`, если окно свёрнуто (размер 0).
    pub fn new(
        ctx: &GpuContext,
        loader: ash::khr::swapchain::Device,
        width: u32,
        height: u32,
        vsync: bool,
        old: vk::SwapchainKHR,
    ) -> Result<Option<Self>> {
        let pd = ctx.physical_device;
        // SAFETY: valid physical device and surface.
        let (caps, formats, modes) = unsafe {
            (
                ctx.surface_fn.get_physical_device_surface_capabilities(pd, ctx.surface)?,
                ctx.surface_fn.get_physical_device_surface_formats(pd, ctx.surface)?,
                ctx.surface_fn.get_physical_device_surface_present_modes(pd, ctx.surface)?,
            )
        };

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        if extent.width == 0 || extent.height == 0 {
            return Ok(None);
        }

        let format = choose_format(&formats);
        let present_mode = choose_present_mode(&modes, vsync);

        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }

        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::INHERIT,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        ]
        .into_iter()
        .find(|mode| caps.supported_composite_alpha.contains(*mode))
        .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE);

        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(ctx.surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old);

        // SAFETY: info references a live surface; old swapchain (if any) is still valid.
        let handle = unsafe { loader.create_swapchain(&info, None) }?;
        // SAFETY: handle was just created.
        let images = unsafe { loader.get_swapchain_images(handle) }?;

        let device = &ctx.device;
        let mut views = Vec::with_capacity(images.len());
        let mut render_finished = Vec::with_capacity(images.len());
        for &image in &images {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(color_range());
            // SAFETY: image belongs to the swapchain created above.
            views.push(unsafe { device.create_image_view(&view_info, None) }?);
            // SAFETY: valid device.
            render_finished
                .push(unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?);
        }

        log::info!(
            "swapchain {}x{}, {} images, format {:?}, present mode {:?}",
            extent.width,
            extent.height,
            images.len(),
            format.format,
            present_mode
        );

        Ok(Some(Self {
            loader,
            handle,
            format,
            extent,
            present_mode,
            images,
            views,
            render_finished,
        }))
    }

    /// Уничтожает swapchain. Вызывать только когда видеокарта его больше не использует.
    pub fn destroy(&mut self, device: &Device) {
        // SAFETY: caller guarantees the GPU is idle with respect to these objects.
        unsafe {
            for view in self.views.drain(..) {
                device.destroy_image_view(view, None);
            }
            for semaphore in self.render_finished.drain(..) {
                device.destroy_semaphore(semaphore, None);
            }
            self.loader.destroy_swapchain(self.handle, None);
        }
        self.handle = vk::SwapchainKHR::null();
    }
}

/// Диапазон «весь цвет, один mip-уровень, один слой» — для картинок swapchain.
pub fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Предпочитаем sRGB-формат: тогда видеокарта сама переводит линейный цвет в цвет экрана.
fn choose_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    let preferred = [vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB];
    preferred
        .iter()
        .find_map(|wanted| {
            formats.iter().copied().find(|f| {
                f.format == *wanted && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
        })
        .unwrap_or(formats[0])
}

fn choose_present_mode(modes: &[vk::PresentModeKHR], vsync: bool) -> vk::PresentModeKHR {
    if vsync {
        return vk::PresentModeKHR::FIFO;
    }
    [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE]
        .into_iter()
        .find(|mode| modes.contains(mode))
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

/// Признак sRGB-формата (нужен отладочному интерфейсу, чтобы правильно смешивать цвета).
pub fn is_srgb(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB | vk::Format::A8B8G8R8_SRGB_PACK32
    )
}

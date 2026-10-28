//! Подключение к Vulkan.
//!
//! Порядок такой:
//! 1. загружаем системную библиотеку Vulkan (`vulkan-1.dll`);
//! 2. создаём экземпляр (instance) — «вход» в Vulkan; при отладке включаем слой проверки ошибок;
//! 3. создаём поверхность (surface) — связь Vulkan с окном игры;
//! 4. выбираем видеокарту, которая умеет Vulkan 1.3, динамический рендеринг, synchronization2
//!    и indirect-отрисовку многими командами за один вызов (её заполняет видеокарта при отсечении);
//! 5. создаём логическое устройство (device) и берём очередь, куда будем отправлять команды.
//!
//! Всё это уничтожается в обратном порядке в `Drop`.

use std::ffi::{CStr, c_char, c_void};

use anyhow::{Context as _, Result, bail};
use ash::{Device, Entry, Instance, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Сведения о выбранной видеокарте (для экрана F3 и журнала).
#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub device_type: vk::PhysicalDeviceType,
    pub api_version: u32,
    pub driver: String,
    /// Сколько наносекунд в одном «тике» счётчика времени видеокарты.
    pub timestamp_period_ns: f32,
    pub timestamps_supported: bool,
    /// Видеокарта сама задаёт число команд отрисовки (`vkCmdDrawIndirectCount`).
    /// Без этого рисуется полный список, а лишние команды пустые.
    pub draw_indirect_count: bool,
    /// Раздельные режимы (layouts) для глубины и трафарета.
    pub separate_depth_stencil_layouts: bool,
}

pub struct GpuContext {
    // Загруженная библиотека Vulkan: напрямую не читается, но должна жить, пока жив экземпляр Vulkan.
    #[allow(dead_code)]
    entry: Entry,
    pub instance: Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub surface_fn: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub queue_family: u32,
    pub queue: vk::Queue,
    pub info: GpuInfo,
    pub validation: bool,
}

impl GpuContext {
    pub fn new(
        display: &impl HasDisplayHandle,
        window: &impl HasWindowHandle,
        want_validation: bool,
    ) -> Result<Self> {
        // SAFETY: loading the system Vulkan loader; no other Vulkan state exists yet.
        let entry = unsafe { Entry::load() }
            .context("Vulkan loader (vulkan-1.dll) not found; update the GPU driver")?;

        // SAFETY: plain query on a valid entry.
        let instance_version = unsafe { entry.try_enumerate_instance_version() }?
            .unwrap_or(vk::API_VERSION_1_0);
        if instance_version < vk::API_VERSION_1_3 {
            bail!(
                "Vulkan 1.3 is required, loader reports {}.{}",
                vk::api_version_major(instance_version),
                vk::api_version_minor(instance_version)
            );
        }

        let raw_display = display
            .display_handle()
            .map_err(|e| anyhow::anyhow!("no display handle: {e:?}"))?
            .as_raw();
        let raw_window = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("no window handle: {e:?}"))?
            .as_raw();

        let mut extensions: Vec<*const c_char> =
            ash_window::enumerate_required_extensions(raw_display)?.to_vec();

        // SAFETY: plain query on a valid entry.
        let layers_available = unsafe { entry.enumerate_instance_layer_properties() }?;
        let validation = want_validation
            && layers_available
                .iter()
                .any(|layer| layer.layer_name_as_c_str() == Ok(VALIDATION_LAYER));
        if want_validation && !validation {
            log::warn!("validation requested but VK_LAYER_KHRONOS_validation is not installed");
        }
        if validation {
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        let layers: Vec<*const c_char> = if validation {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Parodia")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"Parodia")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);
        // SAFETY: all pointers in instance_info live until this call returns.
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .context("vkCreateInstance failed")?;

        let debug = if validation {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            // SAFETY: valid instance with debug_utils enabled.
            let messenger = unsafe { loader.create_debug_utils_messenger(&info, None) }?;
            Some((loader, messenger))
        } else {
            None
        };

        let surface_fn = ash::khr::surface::Instance::new(&entry, &instance);
        // SAFETY: handles come from a live winit window that outlives the renderer.
        let surface =
            unsafe { ash_window::create_surface(&entry, &instance, raw_display, raw_window, None) }
                .context("failed to create window surface")?;

        let (physical_device, queue_family, info) =
            pick_physical_device(&instance, &surface_fn, surface)?;

        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        // Раздельные режимы глубины (DEPTH_ATTACHMENT_OPTIMAL / DEPTH_READ_ONLY_OPTIMAL) включаем,
        // если видеокарта их знает: буфер глубины то рисуется, то читается при построении пирамиды.
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .draw_indirect_count(info.draw_indirect_count)
            .separate_depth_stencil_layouts(info.separate_depth_stencil_layouts);
        let features10 = vk::PhysicalDeviceFeatures::default()
            .multi_draw_indirect(true)
            .draw_indirect_first_instance(true);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(features10)
            .push_next(&mut features12)
            .push_next(&mut features13);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut features);
        // SAFETY: physical_device was checked to support everything requested above.
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .context("vkCreateDevice failed")?;
        // SAFETY: queue family index and queue 0 were requested in device_info.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        Ok(Self {
            entry,
            instance,
            debug,
            surface_fn,
            surface,
            physical_device,
            device,
            queue_family,
            queue,
            info,
            validation,
        })
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        // SAFETY: the renderer destroys every child object before the context is dropped.
        unsafe {
            self.device.destroy_device(None);
            self.surface_fn.destroy_surface(self.surface, None);
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// Выбирает видеокарту: нужна поддержка Vulkan 1.3, swapchain, dynamic rendering, synchronization2,
/// multi-draw indirect с firstInstance и очередь, которая умеет и рисовать, и показывать в окне.
/// Дискретная видеокарта предпочтительнее встроенной.
fn pick_physical_device(
    instance: &Instance,
    surface_fn: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32, GpuInfo)> {
    // SAFETY: valid instance.
    let devices = unsafe { instance.enumerate_physical_devices() }?;
    let mut best: Option<(u32, vk::PhysicalDevice, u32, GpuInfo)> = None;

    for physical_device in devices {
        // SAFETY: physical_device comes from enumerate_physical_devices.
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let name = props
            .device_name_as_c_str()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "unknown".into());

        if props.api_version < vk::API_VERSION_1_3 {
            log::info!("skip GPU '{name}': Vulkan 1.3 not supported");
            continue;
        }

        // SAFETY: valid physical device.
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }?;
        let has_swapchain = extensions
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(ash::khr::swapchain::NAME));
        if !has_swapchain {
            log::info!("skip GPU '{name}': no VK_KHR_swapchain");
            continue;
        }

        let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut features12)
            .push_next(&mut features13);
        // SAFETY: valid physical device; features2 chain points to live structs.
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
        let features10 = features2.features;
        if features13.dynamic_rendering == vk::FALSE || features13.synchronization2 == vk::FALSE {
            log::info!("skip GPU '{name}': no dynamic rendering / synchronization2");
            continue;
        }
        if features10.multi_draw_indirect == vk::FALSE
            || features10.draw_indirect_first_instance == vk::FALSE
        {
            log::info!("skip GPU '{name}': no multiDrawIndirect / drawIndirectFirstInstance");
            continue;
        }
        let draw_indirect_count = features12.draw_indirect_count == vk::TRUE;
        let separate_depth_stencil_layouts = features12.separate_depth_stencil_layouts == vk::TRUE;

        // SAFETY: valid physical device.
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let family = families.iter().enumerate().find_map(|(index, family)| {
            let index = index as u32;
            let graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            // SAFETY: valid physical device, family index and surface.
            let present = unsafe {
                surface_fn.get_physical_device_surface_support(physical_device, index, surface)
            }
            .unwrap_or(false);
            (graphics && present).then_some((index, family.timestamp_valid_bits))
        });
        let Some((queue_family, timestamp_bits)) = family else {
            log::info!("skip GPU '{name}': no graphics+present queue");
            continue;
        };

        let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver_props);
        // SAFETY: valid physical device; chain points to a live struct.
        unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
        let driver_name = driver_props
            .driver_name_as_c_str()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let driver_info = driver_props
            .driver_info_as_c_str()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();

        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 500,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 100,
            _ => 10,
        };

        let info = GpuInfo {
            name,
            device_type: props.device_type,
            api_version: props.api_version,
            driver: format!("{driver_name} {driver_info}").trim().to_string(),
            timestamp_period_ns: props.limits.timestamp_period,
            timestamps_supported: props.limits.timestamp_compute_and_graphics == vk::TRUE
                && timestamp_bits > 0,
            draw_indirect_count,
            separate_depth_stencil_layouts,
        };

        if best.as_ref().is_none_or(|(best_score, ..)| score > *best_score) {
            best = Some((score, physical_device, queue_family, info));
        }
    }

    match best {
        Some((_, physical_device, queue_family, info)) => Ok((physical_device, queue_family, info)),
        None => bail!("no GPU with Vulkan 1.3 + dynamic rendering + synchronization2 + multi-draw indirect found"),
    }
}

/// Сюда слой проверки Vulkan присылает найденные ошибки; мы пишем их в журнал.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the validation layer passes a valid callback struct for the duration of the call.
    let message = unsafe {
        if data.is_null() || (*data).p_message.is_null() {
            String::from("<no message>")
        } else {
            CStr::from_ptr((*data).p_message).to_string_lossy().into_owned()
        }
    };
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!(target: "vulkan", "{message}");
    } else {
        log::warn!(target: "vulkan", "{message}");
    }
    vk::FALSE
}

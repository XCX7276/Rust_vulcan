//! Небо.
//!
//! Один треугольник на весь экран на «бесконечной» глубине (0 при обратной глубине).
//! Рисуется после ландшафта с проверкой глубины «больше или равно», поэтому шейдер неба
//! работает только там, где ничего не нарисовано, — лишней работы почти нет.
//! Цвета и направления солнца и луны берутся из данных кадра (`frame_data.rs`).

use std::io::Cursor;

use anyhow::{Context as _, Result};
use ash::{Device, vk};

const SKY_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sky.vert.spv"));
const SKY_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sky.frag.spv"));

pub struct Sky {
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Sky {
    pub fn new(
        device: &Device,
        frame_layout: vk::DescriptorSetLayout,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<Self> {
        let layouts = [frame_layout];
        // SAFETY: valid device and set layout; destroyed in `destroy`.
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
        }?;
        let pipeline = create_pipeline(device, layout, color_format, depth_format)?;
        Ok(Self { layout, pipeline })
    }

    /// Пересоздаёт конвейер под новый формат картинок.
    pub fn recreate_pipeline(
        &mut self,
        device: &Device,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<()> {
        let pipeline = create_pipeline(device, self.layout, color_format, depth_format)?;
        // SAFETY: called after device_wait_idle when the swapchain is rebuilt.
        unsafe { device.destroy_pipeline(self.pipeline, None) };
        self.pipeline = pipeline;
        Ok(())
    }

    /// Рисует небо (внутри прохода отрисовки, после ландшафта).
    pub fn record(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        frame_set: vk::DescriptorSet,
    ) {
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        // SAFETY: cmd is recording inside a rendering scope with matching formats.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[frame_set],
                &[],
            );
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        // SAFETY: caller waited for the device to be idle.
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

fn create_shader(device: &Device, bytes: &[u8]) -> Result<vk::ShaderModule> {
    let code = ash::util::read_spv(&mut Cursor::new(bytes))?;
    // SAFETY: code is valid SPIR-V produced by glslc at build time.
    Ok(unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None) }?)
}

fn create_pipeline(
    device: &Device,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
    depth_format: vk::Format,
) -> Result<vk::Pipeline> {
    let vert = create_shader(device, SKY_VERT)?;
    let frag = create_shader(device, SKY_FRAG)?;
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // Небо на глубине 0 проходит проверку только там, где глубина осталась 0 (ничего не нарисовано).
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_formats = [color_format];
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(depth_format);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    // SAFETY: all referenced state lives until the call returns.
    let result = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
    // SAFETY: modules are no longer needed once the pipeline is created (or creation failed).
    unsafe {
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
    }
    let pipelines = result
        .map_err(|(_, error)| error)
        .context("sky pipeline creation failed")?;
    Ok(pipelines[0])
}

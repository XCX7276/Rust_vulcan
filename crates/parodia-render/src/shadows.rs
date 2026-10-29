//! Каскадные тени от солнца (или луны).
//!
//! Тень — это «что видит источник света». Мир рисуется в карту глубины с точки зрения солнца
//! (ортогонально, лучи параллельны), а при отрисовке кадра для каждой точки проверяется:
//! ближе ли к солнцу что-то другое. Одной карты на весь обзор мало (вблизи тени были бы
//! размытыми квадратами), поэтому карт 4 — **каскады**: маленькая подробная у камеры
//! и всё крупнее дальше (до `SHADOW_DISTANCE`).
//!
//! - Каскад — шар вокруг своего отрезка обзора камеры (радиус не зависит от поворота камеры,
//!   поэтому размер точки тени постоянен), центр сдвигается по сетке точек карты —
//!   тени не «дрожат» при движении.
//! - **Кеш каскадов**: ближний каскад перерисовывается каждый кадр, дальние — раз в 2, 4 и 8 кадров
//!   (не больше двух каскадов за кадр). Пока каскад не перерисован, он используется с тем
//!   положением, в котором был нарисован, — матрица пересчитывается от текущей камеры.
//! - Карта — массив из 4 слоёв глубины D16 (2 байта на точку).
//! - В карту рисуются стороны блоков, повёрнутые **от** света («задние» для солнца): освещённая
//!   грань тогда не может затенить сама себя, и почти не нужен сдвиг, из-за которого тени
//!   «отъезжают» от предметов.
//! - Тени пиксельные: шейдер берёт тень в центре «пикселя» блока (1/16 блока) по сетке мира, поэтому
//!   края теней — ровные ступеньки из пикселей блока, параллельные граням, а не косая «лесенка»
//!   точек карты.
//! - **Сетка карты по блокам** (`ShadowGrid::Blocks`, по умолчанию). Обычная карта — «квадратная
//!   фотография» с солнца: её строки и столбцы перпендикулярны друг другу. Если солнце не в плоскости
//!   X–Y (склонение пути), рёбра блоков вдоль X и Z на этой фотографии идут не вдоль строк, и край
//!   тени от них получается «лесенкой» из точек карты. Поэтому оси карты берутся косыми: по
//!   столбцу карты меняется только X-координата тени, по строке — только Z (идея — проекция вдоль
//!   лучей, как на рисунке с тенью куба). Тогда тень ребра вдоль X — всегда строка карты, вдоль Z —
//!   столбец: края прямые при любом положении солнца. Наискосок идут только тени вертикальных рёбер
//!   (по направлению лучей) — так и должно быть. `ShadowGrid::Sun` — прежняя квадратная карта
//!   (для сравнения, клавиша G).

use anyhow::{Context as _, Result};
use ash::{Device, vk};
use glam::{DVec3, Mat4, Vec3, Vec4};

use crate::resources::{GpuImage, SharedAllocator};

pub const CASCADES: usize = 4;
/// Дальние границы каскадов (блоков от камеры).
pub const CASCADE_SPLITS: [f32; CASCADES] = [24.0, 64.0, 160.0, 400.0];
/// Дальше этого тени не считаются (последний каскад).
pub const SHADOW_DISTANCE: f32 = 400.0;
/// На сколько блоков к солнцу от каскада ещё могут стоять предметы, отбрасывающие тень в него.
const CASTER_REACH: f32 = 600.0;
pub const SHADOW_FORMAT: vk::Format = vk::Format::D16_UNORM;

/// Как направлены оси карты теней.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowGrid {
    /// Косые оси по рёбрам блоков: тени рёбер вдоль X и Z — прямые строки и столбцы карты.
    Blocks,
    /// Прямоугольные оси «как видит солнце» (как было до решения 60).
    Sun,
}

impl ShadowGrid {
    /// Из флага: `blocks` / `sun` (что угодно другое — `Blocks`).
    pub fn from_flag(value: Option<&str>) -> Self {
        match value.map(str::to_ascii_lowercase).as_deref() {
            Some("sun") => Self::Sun,
            _ => Self::Blocks,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Blocks => "blocks",
            Self::Sun => "sun",
        }
    }

    pub fn toggled(self) -> Self {
        match self {
            Self::Blocks => Self::Sun,
            Self::Sun => Self::Blocks,
        }
    }
}

/// Положение каскада, в котором он был нарисован.
#[derive(Clone, Copy, Debug)]
struct Cascade {
    /// Оси карты: x, y — по карте (единичные, но в режиме `Blocks` не перпендикулярны),
    /// z — от солнца вглубь (перпендикулярна x и y).
    axes: [Vec3; 3],
    /// Координаты центра каскада по осям (x, y — сдвинуты по сетке точек карты).
    offset: [f64; 3],
    radius: f32,
}

/// Камера для расчёта каскадов.
#[derive(Clone, Copy, Debug)]
pub struct ShadowCamera {
    /// Позиция камеры относительно точки отсчёта рендера.
    pub position: DVec3,
    pub forward: Vec3,
    pub fov_y: f32,
    pub aspect: f32,
}

pub struct ShadowCascades {
    size: u32,
    grid: ShadowGrid,
    cascades: [Option<Cascade>; CASCADES],
    /// Направление света, с которым рисовались каскады.
    light: Option<Vec3>,
}

impl ShadowCascades {
    pub fn new(size: u32, grid: ShadowGrid) -> Self {
        Self {
            size,
            grid,
            cascades: [None; CASCADES],
            light: None,
        }
    }

    pub fn grid(&self) -> ShadowGrid {
        self.grid
    }

    /// Меняет оси карты; все каскады перерисуются в следующем кадре.
    pub fn set_grid(&mut self, grid: ShadowGrid) {
        if grid != self.grid {
            self.grid = grid;
            self.light = None;
        }
    }

    /// Какие каскады перерисовать в кадре `frame`: ближний — всегда, дальние — по расписанию
    /// (не больше двух за кадр) или если ещё ни разу не рисовались. Если направление света
    /// сменилось (тени шагают раз в игровую минуту) — все сразу: иначе каскады какое-то время
    /// показывали бы тени разных «шагов», и на их границах тени прыгали бы туда-сюда.
    pub fn due(&self, frame: u64, light: Vec3) -> [bool; CASCADES] {
        if self.light != Some(light) {
            return [true; CASCADES];
        }
        let phase = frame % 8;
        let scheduled = [true, phase.is_multiple_of(2), phase == 1 || phase == 5, phase == 3];
        std::array::from_fn(|i| scheduled[i] || self.cascades[i].is_none())
    }

    /// Ставит каскад `index` в новое положение (перед тем, как его нарисовать).
    pub fn place(&mut self, index: usize, camera: &ShadowCamera, light_dir: Vec3) {
        let near = if index == 0 { 0.0 } else { CASCADE_SPLITS[index - 1] };
        let far = CASCADE_SPLITS[index];
        let (center_distance, radius) = slice_sphere(near, far, camera.fov_y, camera.aspect);
        // Размер точки карты — степень двойки (…, 1/16, 1/8, …, 1 блок), радиус подгоняется под неё.
        // Вместе со сдвигом центра по сетке это значит: границы точек карты вдоль оси Z мира
        // совпадают с границами блоков (и пикселей блока) во всех каскадах — тень не «прыгает»
        // между каскадами и при движении камеры.
        let texel = (2.0 * radius / self.size as f32).log2().ceil().exp2();
        let radius = texel * self.size as f32 * 0.5;
        self.light = Some(light_dir);
        let axes = match self.grid {
            ShadowGrid::Blocks => block_axes(light_dir),
            ShadowGrid::Sun => light_axes(light_dir),
        };
        let center = camera.position + (camera.forward * center_distance).as_dvec3();

        // Сдвиг центра по сетке точек карты (по осям x и y карты).
        let texel = texel as f64;
        let coord = |axis: Vec3| center.dot(axis.as_dvec3());
        let snap = |v: f64| (v / texel).floor() * texel;
        self.cascades[index] = Some(Cascade {
            axes,
            offset: [snap(coord(axes[0])), snap(coord(axes[1])), coord(axes[2])],
            radius,
        });
    }

    /// Матрицы «позиция относительно камеры → пространство карты» для всех каскадов
    /// (для ещё не нарисованных — нулевые). `camera` — текущая позиция камеры.
    pub fn matrices(&self, camera: DVec3) -> [Mat4; CASCADES] {
        std::array::from_fn(|i| match &self.cascades[i] {
            Some(c) => cascade_matrix(c, camera),
            None => Mat4::ZERO,
        })
    }

    /// Размер одной точки карты в блоках, по каскадам.
    pub fn texel_sizes(&self) -> [f32; CASCADES] {
        std::array::from_fn(|i| {
            self.cascades[i].map_or(1.0, |c| 2.0 * c.radius / self.size as f32)
        })
    }
}

/// Шар, в который помещается отрезок обзора [near, far]: (расстояние до центра по взгляду, радиус).
fn slice_sphere(near: f32, far: f32, fov_y: f32, aspect: f32) -> (f32, f32) {
    let tan_v = (fov_y * 0.5).tan();
    let tan_h = tan_v * aspect;
    let spread = (tan_v * tan_v + tan_h * tan_h).sqrt(); // половина диагонали кадра на расстоянии 1
    let (rn, rf) = (near * spread, far * spread);
    // Центр на оси, равноудалённый от углов ближнего и дальнего сечения.
    let center = ((far * far - near * near + rf * rf - rn * rn) / (2.0 * (far - near))).min(far);
    let radius = ((center - near).powi(2) + rn * rn)
        .sqrt()
        .max(((far - center).powi(2) + rf * rf).sqrt());
    (center, radius)
}

/// Оси пространства света: z — направление лучей (от источника), x и y — плоскость карты.
fn light_axes(light_dir: Vec3) -> [Vec3; 3] {
    let z = -light_dir.normalize();
    let helper = if z.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
    let x = helper.cross(z).normalize();
    let y = z.cross(x);
    [x, y, z]
}

/// Косые оси карты (режим `Blocks`): x карты не меняется вдоль рёбер по Z (x ⊥ Z), y карты —
/// вдоль рёбер по X (y ⊥ X); обе перпендикулярны лучам, чтобы предмет и его тень попадали
/// в одну точку карты. Если луч почти вдоль X или Z (у горизонта), рёбра этой оси видны
/// с солнца точками — для них ось берётся просто перпендикулярной другой.
fn block_axes(light_dir: Vec3) -> [Vec3; 3] {
    const TINY: f32 = 1.0e-4;
    let l = light_dir.normalize();
    let a = Vec3::Z.cross(l);
    let b = Vec3::X.cross(l);
    let (x, y) = if a.length() < TINY {
        let y = b.normalize();
        (y.cross(l).normalize(), y)
    } else if b.length() < TINY {
        let x = a.normalize();
        (x, l.cross(x).normalize())
    } else {
        (a.normalize(), b.normalize())
    };
    // Почти параллельные оси карты сжали бы её в линию — тогда прямоугольные оси.
    if x.cross(y).length() < 0.2 {
        return light_axes(light_dir);
    }
    [x, y, -l]
}

fn cascade_matrix(c: &Cascade, camera: DVec3) -> Mat4 {
    // Позиция относительно камеры p → координата по оси: axis·p + (axis·камера − offset).
    // Слагаемые в скобках большие, но их разность маленькая — считаем её в f64.
    let shift = |i: usize| (camera.dot(c.axes[i].as_dvec3()) - c.offset[i]) as f32;
    let [x, y, z] = c.axes;
    let r = c.radius;
    let near = -(r + CASTER_REACH);
    let far = r;
    let depth = 1.0 / (far - near);
    let rows = [
        Vec4::new(x.x / r, x.y / r, x.z / r, shift(0) / r),
        Vec4::new(y.x / r, y.y / r, y.z / r, shift(1) / r),
        Vec4::new(z.x * depth, z.y * depth, z.z * depth, (shift(2) - near) * depth),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    ];
    Mat4::from_cols(rows[0], rows[1], rows[2], rows[3]).transpose()
}

/// Карта теней: массив слоёв глубины (по слою на каскад).
pub struct ShadowMap {
    image: GpuImage,
    /// Вид на каждый слой — для отрисовки в него.
    layer_views: Vec<vk::ImageView>,
    sampler: vk::Sampler,
    pub size: u32,
    /// Слой ещё ни разу не рисовался (его содержимое не определено).
    fresh: [bool; CASCADES],
}

impl ShadowMap {
    pub fn new(device: &Device, allocator: &SharedAllocator, size: u32) -> Result<Self> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(SHADOW_FORMAT)
            .extent(vk::Extent3D {
                width: size,
                height: size,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(CASCADES as u32)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = GpuImage::with_info(
            device,
            allocator,
            &info,
            vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageAspectFlags::DEPTH,
            "shadow map",
        )?;
        let mut layer_views = Vec::with_capacity(CASCADES);
        for layer in 0..CASCADES as u32 {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image.image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(SHADOW_FORMAT)
                .subresource_range(layer_range(layer));
            // SAFETY: the image is bound to memory; views are destroyed in `destroy`.
            layer_views.push(unsafe { device.create_image_view(&view_info, None) }?);
        }
        // Сравнение глубины прямо при чтении: «точка не дальше, чем в карте» = светло.
        // Без сглаживания — края тени чёткие, «пиксельные». За краем карты — белая рамка (светло).
        // SAFETY: valid device.
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                    .compare_enable(true)
                    .compare_op(vk::CompareOp::LESS_OR_EQUAL),
                None,
            )
        }
        .context("failed to create shadow sampler")?;
        Ok(Self {
            image,
            layer_views,
            sampler,
            size,
            fresh: [true; CASCADES],
        })
    }

    pub fn view(&self) -> vk::ImageView {
        self.image.view
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    /// Начинает отрисовку в слой `layer` (очистка в 1 = «далеко»). Вызывать вне другой отрисовки.
    pub fn begin(&mut self, device: &Device, cmd: vk::CommandBuffer, layer: usize) {
        // Прошлое содержимое слоя не нужно — он рисуется заново целиком.
        let to_attachment = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .image(self.image.image)
            .subresource_range(layer_range(layer as u32));
        let depth = vk::RenderingAttachmentInfo::default()
            .image_view(self.layer_views[layer])
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });
        let extent = vk::Extent2D {
            width: self.size,
            height: self.size,
        };
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            })
            .layer_count(1)
            .depth_attachment(&depth);
        // SAFETY: cmd is recording outside a rendering scope; the image and view are alive.
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_attachment]),
            );
            device.cmd_begin_rendering(cmd, &rendering);
        }
        self.fresh[layer] = false;
    }

    /// Заканчивает отрисовку слоя: дальше его читает шейдер ландшафта.
    pub fn end(&self, device: &Device, cmd: vk::CommandBuffer, layer: usize) {
        let to_read = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.image.image)
            .subresource_range(layer_range(layer as u32));
        // SAFETY: cmd is recording inside the rendering scope begun in `begin`.
        unsafe {
            device.cmd_end_rendering(cmd);
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_read]),
            );
        }
    }

    /// Слои, которые ещё ни разу не рисовались (их надо хотя бы очистить до первого чтения).
    pub fn fresh_layers(&self) -> [bool; CASCADES] {
        self.fresh
    }

    pub fn destroy(&mut self, device: &Device, allocator: &SharedAllocator) {
        // SAFETY: caller waited for the device to be idle.
        unsafe {
            for view in self.layer_views.drain(..) {
                device.destroy_image_view(view, None);
            }
            device.destroy_sampler(self.sampler, None);
        }
        self.image.destroy(device, allocator);
    }
}

fn layer_range(layer: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::DEPTH,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: layer,
        layer_count: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(position: DVec3) -> ShadowCamera {
        ShadowCamera {
            position,
            forward: Vec3::new(0.3, -0.2, -0.93).normalize(),
            fov_y: 70f32.to_radians(),
            aspect: 16.0 / 9.0,
        }
    }

    #[test]
    fn slice_sphere_contains_slice_corners() {
        let (fov, aspect) = (70f32.to_radians(), 16.0 / 9.0);
        for (near, far) in [(0.0, 24.0), (24.0, 64.0), (160.0, 400.0)] {
            let (center, radius) = slice_sphere(near, far, fov, aspect);
            let spread = ((fov * 0.5).tan().powi(2) * (1.0 + aspect * aspect)).sqrt();
            for (d, r) in [(near, near * spread), (far, far * spread)] {
                let distance = ((d - center).powi(2) + r * r).sqrt();
                assert!(distance <= radius + 1e-3, "corner at {d} outside sphere");
            }
        }
    }

    #[test]
    fn slice_points_land_inside_cascade_map() {
        let light = Vec3::new(0.4, 0.8, 0.3).normalize();
        let cam = camera(DVec3::new(1234.5, 80.25, -987.75));
        let mut cascades = ShadowCascades::new(2048, ShadowGrid::Blocks);
        for i in 0..CASCADES {
            cascades.place(i, &cam, light);
        }
        let matrices = cascades.matrices(cam.position);
        for (i, m) in matrices.iter().enumerate() {
            // Точка на оси взгляда посередине отрезка каскада.
            let near = if i == 0 { 0.0 } else { CASCADE_SPLITS[i - 1] };
            let p = cam.forward * (near + CASCADE_SPLITS[i]) * 0.5;
            let clip = *m * p.extend(1.0);
            assert!(clip.x.abs() < 1.0 && clip.y.abs() < 1.0, "cascade {i}: {clip:?}");
            assert!(clip.z > 0.0 && clip.z < 1.0, "cascade {i} depth {}", clip.z);
        }
    }

    #[test]
    fn same_world_point_same_map_texel_after_camera_moves() {
        // Сдвиг камеры меньше точки карты не должен двигать тени (сетка центра).
        let light = Vec3::new(0.4, 0.8, 0.3).normalize();
        let mut a = ShadowCascades::new(1024, ShadowGrid::Blocks);
        let mut b = ShadowCascades::new(1024, ShadowGrid::Blocks);
        let cam_a = camera(DVec3::new(10.0, 70.0, 10.0));
        let cam_b = camera(DVec3::new(10.01, 70.0, 10.0));
        a.place(3, &cam_a, light);
        b.place(3, &cam_b, light);
        let world = DVec3::new(30.0, 60.0, -40.0);
        let ma = a.matrices(cam_a.position)[3] * (world - cam_a.position).as_vec3().extend(1.0);
        let mb = b.matrices(cam_b.position)[3] * (world - cam_b.position).as_vec3().extend(1.0);
        assert!((ma - mb).length() < 1e-3, "{ma:?} vs {mb:?}");
    }

    #[test]
    fn texel_is_power_of_two() {
        let mut cascades = ShadowCascades::new(2048, ShadowGrid::Blocks);
        let cam = camera(DVec3::new(5.5, 70.0, -3.25));
        for i in 0..CASCADES {
            cascades.place(i, &cam, Vec3::new(0.6, 0.8, 0.0));
        }
        for texel in cascades.texel_sizes() {
            assert_eq!(texel.log2().fract(), 0.0, "texel {texel}");
        }
    }

    #[test]
    fn slice_points_land_inside_cascade_map_in_both_grids() {
        let lights = [
            Vec3::new(0.4, 0.8, 0.3),
            Vec3::new(0.7, 0.5, 0.5),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.99, 0.05, 0.05),
            Vec3::new(0.0, 0.5, 0.87),
            Vec3::Y,
        ];
        for grid in [ShadowGrid::Blocks, ShadowGrid::Sun] {
            for light in lights {
                let light = light.normalize();
                let cam = camera(DVec3::new(-77.5, 64.0, 301.25));
                let mut cascades = ShadowCascades::new(1024, grid);
                for i in 0..CASCADES {
                    cascades.place(i, &cam, light);
                }
                let texels = cascades.texel_sizes();
                for (i, m) in cascades.matrices(cam.position).iter().enumerate() {
                    assert!(m.is_finite(), "{grid:?} {light} cascade {i}");
                    // Карта не сплющена: определитель без масштаба ≈ синус угла между осями карты.
                    let r = texels[i] * 1024.0 * 0.5;
                    let unscaled = m.determinant() * r * r * (2.0 * r + CASTER_REACH);
                    assert!(unscaled.abs() > 0.2, "{grid:?} {light} cascade {i}: {unscaled}");
                    // Точки на краю шара каскада — внутри карты.
                    let (center, radius) = slice_sphere(
                        if i == 0 { 0.0 } else { CASCADE_SPLITS[i - 1] },
                        CASCADE_SPLITS[i],
                        cam.fov_y,
                        cam.aspect,
                    );
                    for dir in [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z] {
                        let p = cam.forward * center + dir * radius * 0.99;
                        let clip = *m * p.extend(1.0);
                        assert!(clip.x.abs() <= 1.0 && clip.y.abs() <= 1.0, "{grid:?} {light} {i}: {clip:?}");
                        assert!(clip.z > 0.0 && clip.z < 1.0);
                    }
                }
            }
        }
    }

    #[test]
    fn block_grid_keeps_block_edges_on_map_rows_and_columns() {
        // Склонение 45°: солнце не в плоскости X–Y.
        let light = Vec3::new(0.7, 0.5, 0.5).normalize();
        let cam = camera(DVec3::new(3.0, 70.0, 4.0));
        let mut cascades = ShadowCascades::new(1024, ShadowGrid::Blocks);
        cascades.place(0, &cam, light);
        let m = cascades.matrices(cam.position)[0];
        let map = |p: Vec3| m * p.extend(1.0);
        let corner = Vec3::new(2.0, -3.0, -5.0);
        // Ребро вдоль X — одна строка карты (y не меняется), вдоль Z — один столбец (x не меняется).
        assert!((map(corner).y - map(corner + Vec3::X * 7.0).y).abs() < 1e-5);
        assert!((map(corner).x - map(corner + Vec3::Z * 7.0).x).abs() < 1e-5);
        // Точка и её «тень» вдоль луча — в одной точке карты.
        let along = map(corner + light * 5.0);
        assert!((map(corner).x - along.x).abs() < 1e-5 && (map(corner).y - along.y).abs() < 1e-5);
        // В квадратной карте ребро вдоль X при таком солнце идёт наискосок.
        let mut sun = ShadowCascades::new(1024, ShadowGrid::Sun);
        sun.place(0, &cam, light);
        let m = sun.matrices(cam.position)[0];
        let (a, b) = (m * corner.extend(1.0), m * (corner + Vec3::X * 7.0).extend(1.0));
        assert!((a.x - b.x).abs() > 1e-3 && (a.y - b.y).abs() > 1e-3);
    }

    #[test]
    fn changing_grid_redraws_everything() {
        let mut cascades = ShadowCascades::new(512, ShadowGrid::Blocks);
        let cam = camera(DVec3::ZERO);
        for i in 0..CASCADES {
            cascades.place(i, &cam, Vec3::Y);
        }
        assert_ne!(cascades.due(1, Vec3::Y), [true; CASCADES]);
        cascades.set_grid(ShadowGrid::Sun);
        assert_eq!(cascades.due(1, Vec3::Y), [true; CASCADES]);
    }

    #[test]
    fn far_cascades_update_less_often() {
        let mut cascades = ShadowCascades::new(512, ShadowGrid::Blocks);
        let cam = camera(DVec3::ZERO);
        for i in 0..CASCADES {
            cascades.place(i, &cam, Vec3::Y);
        }
        let mut counts = [0; CASCADES];
        for frame in 0..64 {
            let due = cascades.due(frame, Vec3::Y);
            assert!(due.iter().filter(|d| **d).count() <= 2);
            for i in 0..CASCADES {
                counts[i] += due[i] as u32;
            }
        }
        assert_eq!(counts, [64, 32, 16, 8]);
        // Свет сменился — все каскады сразу.
        assert_eq!(cascades.due(1, Vec3::X), [true; CASCADES]);
    }
}

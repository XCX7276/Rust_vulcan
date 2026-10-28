//! Летающая камера (свободный полёт сквозь мир).
//!
//! Позиция хранится в `f64`: точности хватает и в миллионе блоков от центра мира.
//! В рендер передаётся «целая часть + дробная часть», а сама камера для видеокарты всегда
//! стоит в начале координат — так в шейдере не теряется точность вдали от нуля.
//!
//! Углы: `yaw` — поворот влево-вправо (0 = взгляд на −Z), `pitch` — вверх-вниз.
//! Эта камера останется для режима полёта и отладки; камера игрока появится на этапе 8.

use std::collections::HashSet;

use glam::{DVec3, Mat4, Vec3};
use winit::keyboard::KeyCode;

/// Блоков в секунду.
const SPEED: f64 = 20.0;
/// Во сколько раз быстрее с зажатым Shift.
const FAST_MULTIPLIER: f64 = 6.0;
/// Радиан на пиксель движения мыши.
const MOUSE_SENSITIVITY: f32 = 0.0025;
/// Ближняя плоскость отсечения (м). Дальней нет — проекция бесконечная.
const NEAR: f32 = 0.05;

pub struct FlyCamera {
    pub position: DVec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_degrees: f32,
}

impl FlyCamera {
    pub fn new(position: DVec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: -0.35,
            fov_y_degrees: 70.0,
        }
    }

    /// Куда смотрит камера (единичный вектор).
    pub fn forward(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        Vec3::new(sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch)
    }

    /// Вектор «вправо» в горизонтальной плоскости.
    pub fn right(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        Vec3::new(cos_yaw, 0.0, sin_yaw)
    }

    /// Поворот от движения мыши (в пикселях).
    pub fn rotate(&mut self, dx: f64, dy: f64) {
        self.yaw += dx as f32 * MOUSE_SENSITIVITY;
        self.pitch = (self.pitch - dy as f32 * MOUSE_SENSITIVITY).clamp(-1.55, 1.55);
        self.yaw = self.yaw.rem_euclid(std::f32::consts::TAU);
    }

    /// Движение по нажатым клавишам за время `dt` (секунды).
    /// W/S — вперёд/назад по взгляду, A/D — вбок, Пробел/Ctrl — вверх/вниз, Shift — быстрее.
    pub fn update(&mut self, dt: f64, keys: &HashSet<KeyCode>) {
        let pressed = |code| keys.contains(&code);
        let forward = self.forward().as_dvec3();
        let right = self.right().as_dvec3();
        let mut direction = DVec3::ZERO;
        if pressed(KeyCode::KeyW) {
            direction += forward;
        }
        if pressed(KeyCode::KeyS) {
            direction -= forward;
        }
        if pressed(KeyCode::KeyD) {
            direction += right;
        }
        if pressed(KeyCode::KeyA) {
            direction -= right;
        }
        if pressed(KeyCode::Space) {
            direction += DVec3::Y;
        }
        if pressed(KeyCode::ControlLeft) {
            direction -= DVec3::Y;
        }
        if direction.length_squared() > 0.0 {
            let fast = pressed(KeyCode::ShiftLeft) || pressed(KeyCode::ShiftRight);
            let speed = if fast { SPEED * FAST_MULTIPLIER } else { SPEED };
            self.position += direction.normalize() * speed * dt;
        }
    }

    /// Матрица «поворот × проекция» для камеры в начале координат.
    /// Проекция бесконечная, с обратной глубиной (ближе = больше) и перевёрнутой осью Y под Vulkan.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = glam::camera::rh::view::look_to_mat4(Vec3::ZERO, self.forward(), Vec3::Y);
        // Готовая проекция под соглашения Vulkan: глубина 0..1, ось Y вниз.
        let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
            self.fov_y_degrees.to_radians(),
            aspect,
            NEAR,
        );
        projection * view
    }

    /// Блок, в котором стоит камера, и дробная часть позиции.
    pub fn split_position(&self) -> ([i64; 3], [f32; 3]) {
        let floor = self.position.floor();
        let frac = self.position - floor;
        (
            [floor.x as i64, floor.y as i64, floor.z as i64],
            [frac.x as f32, frac.y as f32, frac.z as f32],
        )
    }
}

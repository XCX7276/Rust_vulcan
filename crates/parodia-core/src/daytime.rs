//! Время суток: где солнце и луна, какого цвета свет и небо.
//!
//! Всё считается только из часа (0–24), без случайности: один и тот же час всегда даёт
//! одну и ту же картинку (это понадобится и для сетевой игры — время присылает сервер).
//!
//! - Солнце встаёт на востоке (+X) в 6:00, в полдень стоит в зените,
//!   садится на западе (−X) в 18:00. Луна — напротив солнца.
//! - Свет даёт солнце, пока оно над горизонтом, иначе луна (слабее и холоднее). У горизонта
//!   яркость обоих уходит в ноль, поэтому смена источника (и направления теней) не видна.
//! - Тени двигаются **шагами**: их направление (`shadow_dir`) берётся по времени, округлённому
//!   до игровой минуты (солнце сдвигается на 0,25°). Между шагами тени стоят на месте, а не
//!   «плывут»; за шаг край тени сдвигается меньше чем на пиксель блока.
//! - Небо: цвет у горизонта и в зените плавно меняется день → закат → ночь.
//! - Рассеянный свет («без мрачности»): даже ночью и в тени не опускается до чёрного.
//!
//! Цвета линейные (до перевода в sRGB при выводе на экран).

/// Часы в сутках.
pub const HOURS_PER_DAY: f64 = 24.0;
/// Шаг направления теней — игровая минута.
const SHADOW_STEP_HOURS: f64 = 1.0 / 60.0;

// Путь солнца и луны. По умолчанию (склонение 0) они ходят ровно в плоскости X–Y через зенит.
// Тогда направление света всегда перпендикулярно оси Z, и тени от всех рёбер блоков (вдоль X, Y
// и Z) ложатся на землю линиями вдоль X и Z — параллельно граням блоков.
// Склонение наклоняет путь к +Z: в полдень солнце стоит ниже зенита (на 90° − склонение),
// но тени от вертикальных рёбер идут наискосок «лесенкой».

/// Часы суток с ходом времени.
#[derive(Clone, Copy, Debug)]
pub struct DayClock {
    /// Текущее время, часы (0–24).
    hours: f64,
    /// Сколько реальных минут длятся сутки (0 — время стоит).
    minutes_per_day: f64,
    /// Ускорение хода времени (0 — пауза); меняется клавишами для проверки света.
    speed: f64,
    /// Ускорение до паузы (чтобы снять паузу с тем же ускорением).
    speed_before_pause: f64,
    /// Склонение пути солнца к +Z (радианы, 0 — через зенит).
    tilt: f32,
}

/// Ступени ускорения времени.
const SPEEDS: [f64; 8] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];

impl DayClock {
    pub fn new(start_hours: f64, minutes_per_day: f64) -> Self {
        Self {
            hours: start_hours.rem_euclid(HOURS_PER_DAY),
            minutes_per_day: minutes_per_day.max(0.0),
            speed: 1.0,
            speed_before_pause: 1.0,
            tilt: 0.0,
        }
    }

    /// Задаёт склонение пути солнца и луны (градусы, 0–60).
    pub fn set_tilt_degrees(&mut self, degrees: f32) {
        self.tilt = degrees.clamp(0.0, 60.0).to_radians();
    }

    /// Меняет склонение на `degrees` градусов (в пределах 0–60).
    pub fn change_tilt(&mut self, degrees: f32) {
        self.set_tilt_degrees((self.tilt_degrees() + degrees).round());
    }

    /// Склонение пути солнца и луны, градусы.
    pub fn tilt_degrees(&self) -> f32 {
        self.tilt.to_degrees()
    }

    /// Сдвигает время на `hours` часов (солнце и луна переходят по небу выше или ниже).
    pub fn shift(&mut self, hours: f64) {
        self.hours = (self.hours + hours).rem_euclid(HOURS_PER_DAY);
    }

    /// Следующая ступень ускорения (`faster`) или предыдущая; из паузы — ход ×1.
    pub fn change_speed(&mut self, faster: bool) {
        if self.speed == 0.0 {
            self.speed = 1.0;
            return;
        }
        let index = SPEEDS.iter().position(|s| *s >= self.speed).unwrap_or(0);
        let next = if faster {
            (index + 1).min(SPEEDS.len() - 1)
        } else {
            index.saturating_sub(1)
        };
        self.speed = SPEEDS[next];
    }

    /// Пауза / продолжить с прежним ускорением.
    pub fn toggle_pause(&mut self) {
        if self.speed == 0.0 {
            self.speed = self.speed_before_pause;
        } else {
            self.speed_before_pause = self.speed;
            self.speed = 0.0;
        }
    }

    /// Ускорение хода времени (0 — пауза).
    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// Двигает время на `seconds` реальных секунд.
    pub fn advance(&mut self, seconds: f64) {
        if self.minutes_per_day > 0.0 {
            let hours_per_second = HOURS_PER_DAY / (self.minutes_per_day * 60.0) * self.speed;
            self.hours = (self.hours + seconds * hours_per_second).rem_euclid(HOURS_PER_DAY);
        }
    }

    pub fn hours(&self) -> f64 {
        self.hours
    }

    pub fn daylight(&self) -> Daylight {
        Daylight::at_tilted(self.hours, self.tilt)
    }
}

/// Свет и небо в данный час.
#[derive(Clone, Copy, Debug)]
pub struct Daylight {
    /// Направление на солнце и на луну (единичные).
    pub sun_dir: [f32; 3],
    pub moon_dir: [f32; 3],
    /// Направление на источник прямого света (солнце днём, луна ночью).
    pub light_dir: [f32; 3],
    /// То же, но по времени, округлённому до игровой минуты, — по нему строятся тени.
    pub shadow_dir: [f32; 3],
    /// Цвет × яркость прямого света.
    pub light_color: [f32; 3],
    /// Рассеянный свет сверху (от неба) и снизу (отражённый от земли).
    pub ambient_sky: [f32; 3],
    pub ambient_ground: [f32; 3],
    /// Небо: в зените и у горизонта.
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    /// Свечение вокруг солнца у горизонта (закат, рассвет).
    pub sun_glow: [f32; 3],
    /// 0 — ночь, 1 — день.
    pub day: f32,
}

impl Daylight {
    /// Свет в час `hours` без склонения (путь через зенит).
    pub fn at(hours: f64) -> Self {
        Self::at_tilted(hours, 0.0)
    }

    /// Свет в час `hours` при склонении пути `tilt` (радианы).
    pub fn at_tilted(hours: f64, tilt: f32) -> Self {
        let sun_dir = sun_direction(hours, tilt);
        let moon_dir = [-sun_dir[0], -sun_dir[1], -sun_dir[2]];
        let elevation = sun_dir[1];
        let stepped = sun_direction((hours / SHADOW_STEP_HOURS).floor() * SHADOW_STEP_HOURS, tilt);
        let shadow_dir = if stepped[1] >= 0.0 {
            stepped
        } else {
            [-stepped[0], -stepped[1], -stepped[2]]
        };

        let day = smoothstep(-0.15, 0.2, elevation);
        // Сумерки: сильнее всего, когда солнце у самого горизонта.
        let twilight = (-(elevation / 0.12).powi(2)).exp();

        let sun_strength = smoothstep(0.0, 0.15, elevation);
        let moon_strength = smoothstep(0.0, 0.15, -elevation);
        let sun_color = mix3([1.0, 0.55, 0.30], [1.0, 0.95, 0.88], smoothstep(0.0, 0.35, elevation));
        let (light_dir, light_color) = if elevation >= 0.0 {
            (sun_dir, scale3(sun_color, 1.05 * sun_strength))
        } else {
            (moon_dir, scale3([0.55, 0.62, 0.85], 0.16 * moon_strength))
        };

        let sky_zenith = mix3([0.004, 0.006, 0.020], [0.10, 0.27, 0.65], day);
        let mut sky_horizon = mix3([0.010, 0.015, 0.035], [0.55, 0.70, 0.90], day);
        sky_horizon = mix3(sky_horizon, [0.85, 0.45, 0.25], twilight * 0.55);
        let ambient_sky = mix3([0.045, 0.05, 0.075], [0.34, 0.40, 0.50], day);
        let ambient_ground = mix3([0.02, 0.02, 0.028], [0.20, 0.18, 0.14], day);
        let sun_glow = scale3([1.0, 0.45, 0.18], twilight * 0.8);

        Self {
            sun_dir,
            moon_dir,
            light_dir,
            shadow_dir,
            light_color,
            ambient_sky,
            ambient_ground,
            sky_zenith,
            sky_horizon,
            sun_glow,
            day,
        }
    }
}

/// Направление на солнце в час `hours` при склонении `tilt` (радианы).
fn sun_direction(hours: f64, tilt: f32) -> [f32; 3] {
    let phase = ((hours - 6.0) / HOURS_PER_DAY * std::f64::consts::TAU) as f32;
    let (sin_p, cos_p) = phase.sin_cos();
    if tilt == 0.0 {
        return [cos_p, sin_p, 0.0];
    }
    [cos_p, sin_p * tilt.cos(), sin_p * tilt.sin()]
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn luminance(c: [f32; 3]) -> f32 {
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    }

    #[test]
    fn sun_rises_east_peaks_at_noon_sets_west() {
        let dawn = Daylight::at(6.0).sun_dir;
        let noon = Daylight::at(12.0).sun_dir;
        let dusk = Daylight::at(18.0).sun_dir;
        assert!(dawn[0] > 0.99 && dawn[1].abs() < 1e-5);
        assert!(noon[1] > 0.85);
        assert!(dusk[0] < -0.99);
        assert!(Daylight::at(0.0).sun_dir[1] < -0.85);
    }

    #[test]
    fn day_is_bright_night_is_dim_but_not_black() {
        let noon = Daylight::at(12.0);
        let midnight = Daylight::at(0.0);
        assert!(luminance(noon.light_color) > 0.8);
        assert!(luminance(midnight.light_color) < 0.2);
        assert!(luminance(midnight.ambient_sky) > 0.02, "night must not be pitch black");
        // Ночью свет идёт от луны, днём — от солнца.
        assert_eq!(midnight.light_dir, midnight.moon_dir);
        assert_eq!(noon.light_dir, noon.sun_dir);
    }

    #[test]
    fn light_fades_out_at_horizon_so_switch_is_invisible() {
        for hours in [5.999, 6.001, 17.999, 18.001] {
            assert!(luminance(Daylight::at(hours).light_color) < 0.02, "at {hours}");
        }
    }

    #[test]
    fn light_never_tilts_towards_z() {
        // Иначе тени перестают быть параллельны граням блоков.
        for minute in 0..24 * 60 {
            let light = Daylight::at(minute as f64 / 60.0);
            assert_eq!(light.light_dir[2], 0.0);
            assert_eq!(light.shadow_dir[2], 0.0);
        }
    }

    #[test]
    fn tilt_lowers_noon_sun() {
        let noon = Daylight::at_tilted(12.0, 30f32.to_radians());
        assert!((noon.sun_dir[1] - 30f32.to_radians().cos()).abs() < 1e-5);
        assert!(noon.sun_dir[2] > 0.49);
        // Восход и закат на месте.
        assert!(Daylight::at_tilted(6.0, 0.5).sun_dir[0] > 0.99);
    }

    #[test]
    fn shadows_move_in_steps() {
        // В пределах одной игровой минуты направление теней не меняется, а свет меняется.
        let a = Daylight::at(10.0 + 0.1 / 60.0);
        let b = Daylight::at(10.0 + 0.9 / 60.0);
        assert_eq!(a.shadow_dir, b.shadow_dir);
        assert_ne!(a.light_dir, b.light_dir);
        let next = Daylight::at(10.0 + 1.1 / 60.0);
        assert_ne!(a.shadow_dir, next.shadow_dir);
    }

    #[test]
    fn clock_advances_and_wraps() {
        let mut clock = DayClock::new(23.0, 24.0); // сутки = 24 минуты → 1 час = 60 с
        clock.advance(120.0);
        assert!((clock.hours() - 1.0).abs() < 1e-9);
        let mut stopped = DayClock::new(10.0, 0.0);
        stopped.advance(1000.0);
        assert_eq!(stopped.hours(), 10.0);
    }

    #[test]
    fn speed_pause_and_shift() {
        let mut clock = DayClock::new(12.0, 24.0); // ×1: 1 час = 60 с
        clock.change_speed(true);
        clock.change_speed(true);
        assert_eq!(clock.speed(), 4.0);
        clock.advance(15.0);
        assert!((clock.hours() - 13.0).abs() < 1e-9);
        clock.toggle_pause();
        clock.advance(100.0);
        assert!((clock.hours() - 13.0).abs() < 1e-9);
        clock.toggle_pause();
        assert_eq!(clock.speed(), 4.0);
        clock.shift(-14.0);
        assert!((clock.hours() - 23.0).abs() < 1e-9);
    }
}

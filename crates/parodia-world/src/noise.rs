//! Детерминированный шум для генерации мира.
//!
//! Шум — плавно меняющееся «случайное» число для каждой точки. Одно и то же зерно мира и одни
//! и те же координаты всегда дают один и тот же результат — поэтому чанк можно сгенерировать
//! в любом порядке и в любом потоке, и он совпадёт с соседями на границе.
//!
//! - `hash` — превращает (зерно, x, z) в псевдослучайное 64-битное число;
//! - `value_noise_2d` — значения в узлах целочисленной сетки, между узлами — плавная интерполяция;
//! - `fbm_2d` — сумма нескольких «октав» шума разного масштаба: крупные холмы + мелкие неровности.

/// Перемешивание битов (вариант SplitMix64): маленькое изменение входа сильно меняет выход.
#[inline]
pub fn mix64(mut v: u64) -> u64 {
    v ^= v >> 30;
    v = v.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v ^= v >> 27;
    v = v.wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

/// Псевдослучайное число для узла сетки (x, z) при данном зерне.
#[inline]
pub fn hash2(seed: u64, x: i64, z: i64) -> u64 {
    mix64(seed ^ mix64((x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ mix64(z as u64)))
}

/// Число в диапазоне [-1, 1] для узла сетки.
#[inline]
fn lattice(seed: u64, x: i64, z: i64) -> f64 {
    (hash2(seed, x, z) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// Плавная кривая 6t⁵ − 15t⁴ + 10t³: убирает «изломы» на границах клеток сетки.
#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Шум значений в точке (x, z); результат в [-1, 1]. Масштаб — 1 клетка сетки на единицу.
pub fn value_noise_2d(seed: u64, x: f64, z: f64) -> f64 {
    let x0 = x.floor();
    let z0 = z.floor();
    let tx = fade(x - x0);
    let tz = fade(z - z0);
    let (ix, iz) = (x0 as i64, z0 as i64);
    let a = lattice(seed, ix, iz);
    let b = lattice(seed, ix + 1, iz);
    let c = lattice(seed, ix, iz + 1);
    let d = lattice(seed, ix + 1, iz + 1);
    let top = a + (b - a) * tx;
    let bottom = c + (d - c) * tx;
    top + (bottom - top) * tz
}

/// Фрактальный шум: `octaves` слоёв, каждый следующий вдвое мельче и в `gain` раз слабее.
/// Результат примерно в [-1, 1].
pub fn fbm_2d(seed: u64, x: f64, z: f64, octaves: u32, gain: f64) -> f64 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut norm = 0.0;
    for octave in 0..octaves {
        let octave_seed = mix64(seed.wrapping_add(octave as u64));
        sum += value_noise_2d(octave_seed, x * frequency, z * frequency) * amplitude;
        norm += amplitude;
        amplitude *= gain;
        frequency *= 2.0;
    }
    sum / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_seed_dependent() {
        let a = fbm_2d(1337, 12.3, -45.6, 5, 0.5);
        assert_eq!(a, fbm_2d(1337, 12.3, -45.6, 5, 0.5));
        assert_ne!(a, fbm_2d(1338, 12.3, -45.6, 5, 0.5));
    }

    #[test]
    fn range_and_continuity() {
        let mut previous = value_noise_2d(7, 0.0, 3.3);
        for i in 1..10_000 {
            let x = i as f64 * 0.01;
            let v = value_noise_2d(7, x, 3.3);
            assert!((-1.0..=1.0).contains(&v));
            // Соседние точки на расстоянии 0.01 не прыгают.
            assert!((v - previous).abs() < 0.1, "jump at {x}");
            previous = v;
        }
    }

    #[test]
    fn far_coordinates_work() {
        let v = fbm_2d(42, 1_000_000.5, -1_000_000.25, 4, 0.5);
        assert!(v.is_finite() && (-1.0..=1.0).contains(&v));
    }
}

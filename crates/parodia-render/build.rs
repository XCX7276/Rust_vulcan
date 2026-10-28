// Компиляция шейдеров при сборке.
//
// Исходники GLSL лежат в папке `shaders/` в корне проекта. При каждой сборке, если какой-то шейдер
// изменился, он компилируется в SPIR-V (формат, который понимает видеокарта) программой `glslc`
// из Vulkan SDK. Результат кладётся во временную папку сборки и встраивается в exe.
// Ошибка в шейдере = ошибка сборки (сразу видно, а не при запуске игры).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let shader_dir = manifest.join("../../shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!("cargo:rerun-if-changed={}", shader_dir.display());
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let glslc = find_glslc();
    let entries = std::fs::read_dir(&shader_dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", shader_dir.display()));
    for entry in entries {
        let path = entry.expect("shader dir entry").path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !matches!(ext, "vert" | "frag" | "comp") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path.file_name().expect("file name").to_string_lossy();
        let output = out_dir.join(format!("{name}.spv"));
        let result = Command::new(&glslc)
            .arg("--target-env=vulkan1.3")
            .arg("-O")
            .arg("-o")
            .arg(&output)
            .arg(&path)
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "failed to run glslc ({}): {e}. Install the Vulkan SDK (LunarG) and restart the terminal.",
                    glslc.display()
                )
            });
        if !result.status.success() {
            panic!(
                "shader compilation failed: {}\n{}",
                path.display(),
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}

fn find_glslc() -> PathBuf {
    let exe = if cfg!(windows) { "glslc.exe" } else { "glslc" };
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        for sub in ["Bin", "bin"] {
            let candidate = Path::new(&sdk).join(sub).join(exe);
            if candidate.exists() {
                return candidate;
            }
        }
    }
    PathBuf::from(exe)
}

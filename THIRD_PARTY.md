# Сторонние библиотеки

Прямые зависимости (их собственные зависимости подтягиваются автоматически; полный список — `Cargo.lock`).

| Библиотека | Версия | Лицензия | Зачем |
|---|---|---|---|
| ash | 0.38 | MIT / Apache-2.0 | вызовы Vulkan |
| ash-window | 0.13 | MIT / Apache-2.0 | поверхность Vulkan для окна |
| raw-window-handle | 0.6 | MIT / Apache-2.0 / Zlib | общий формат «ручки» окна |
| winit | 0.30 | Apache-2.0 | окно и ввод |
| gpu-allocator | 0.28 | MIT / Apache-2.0 | выделение видеопамяти |
| egui | 0.36 | MIT / Apache-2.0 | отладочный интерфейс F3 |
| egui-winit | 0.36 | MIT / Apache-2.0 | ввод для egui |
| egui-ash-renderer | 0.13 | MIT / Apache-2.0 | отрисовка egui через Vulkan |
| log | 0.4 | MIT / Apache-2.0 | журнал |
| anyhow | 1 | MIT / Apache-2.0 | ошибки |
| serde_json | 1 | MIT / Apache-2.0 | чтение JSON (переводы, блоки) |
| glam | 0.33 | MIT / Apache-2.0 | векторы и матрицы (камера) |
| bytemuck | 1.25 | MIT / Apache-2.0 / Zlib | безопасное представление данных как байтов (загрузка в видеопамять) |

Идея алгоритма binary greedy meshing — из открытого проекта cgerikj/binary-greedy-meshing;
код в `parodia-mesh` написан заново.

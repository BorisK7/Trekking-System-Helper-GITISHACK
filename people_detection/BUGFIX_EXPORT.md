# Исправление проблемы с экспортом аналитики

## Проблема
После рефакторинга перестали сохраняться результаты детекции в папку `analytics` (JSON и CSV файлы).

## Причина
Модуль `exporter.py` использовал устаревший локальный метод `_unpack_track()` для распаковки данных треков. После рефакторинга эта функция была вынесена в общий модуль `utils/common.py`, но `exporter.py` не был обновлен.

## Исправление
1. Добавлен импорт `from .common import unpack_track` в `exporter.py`
2. Удален дублирующий метод `_unpack_track()`
3. Все вызовы заменены на использование общей функции `unpack_track()`

## Как проверить, что экспорт работает

### 1. Запуск с экспортом данных
```bash
python main.py --source input/video.mp4 --export-data --classes person
```

### 2. Проверка результатов
После обработки видео в папке `output/run_YYYYMMDD_HHMMSS/analytics/` должны появиться файлы:
- `detection_analysis_trajectories.json` - траектории в JSON
- `detection_analysis_trajectories.csv` - траектории в CSV
- `detection_analysis_frames.json` - покадровые данные
- `detection_analysis_stats.json` - общая статистика
- `detection_analysis_uwb_format.csv` - формат для сравнения с UWB
- `summary.txt` - текстовый отчет

### 3. Пример вывода в логах
```
2025-11-23 22:30:15 - INFO - Экспорт данных: ВКЛЮЧЕН (в output/run_20251123_223015/analytics)
...
2025-11-23 22:31:45 - INFO - 
Экспорт данных трекинга...

2025-11-23 22:31:46 - INFO - 
Экспортированные файлы:
  trajectories_json: output/run_20251123_223015/analytics/detection_analysis_trajectories.json
  trajectories_csv: output/run_20251123_223015/analytics/detection_analysis_trajectories.csv
  frame_data: output/run_20251123_223015/analytics/detection_analysis_frames.json
  statistics: output/run_20251123_223015/analytics/detection_analysis_stats.json
  uwb_format: output/run_20251123_223015/analytics/detection_analysis_uwb_format.csv
  summary: output/run_20251123_223015/analytics/summary.txt
```

## Дополнительные параметры экспорта

### Экспорт с адаптивным трекингом
```bash
python main.py --source input/video.mp4 --export-data --adaptive-tracking --classes person
```

### Экспорт с распознаванием действий
```bash
python main.py --source input/video.mp4 --export-data --action-recognition --pose-detection --classes person
```

### Экспорт всех объектов (не только людей)
```bash
python main.py --source input/video.mp4 --export-data --classes all
```

## Структура экспортируемых данных

### trajectories.json
```json
{
  "export_date": "2025-11-23T22:30:15",
  "total_tracks": 5,
  "trajectories": [
    {
      "track_id": 1,
      "class": "person",
      "points": [
        {"x": 640.5, "y": 480.2, "timestamp": 0.033, "proj_x": 960, "proj_y": 540},
        ...
      ],
      "metadata": {
        "duration": 12.5,
        "total_distance": 450.3,
        "avg_speed": 36.0,
        "final_state": "DYNAMIC"
      }
    }
  ]
}
```

### trajectories.csv
```csv
track_id,class,x,y,timestamp,proj_x,proj_y
1,person,640.5,480.2,0.033,960,540
1,person,642.1,481.5,0.066,963,542
...
```

## Файлы, затронутые исправлением
- `people_detection/utils/exporter.py` - основное исправление
- `people_detection/utils/common.py` - общая функция unpack_track
- `people_detection/utils/analytics.py` - обновлен для использования common.py
- `people_detection/utils/action_recognition.py` - обновлен для использования geometry.py
- `people_detection/utils/visualizer.py` - обновлен для использования common.py

## Дата исправления
23 ноября 2025


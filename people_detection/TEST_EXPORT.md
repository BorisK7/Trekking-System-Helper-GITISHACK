# Тестирование экспорта данных

## Быстрый тест

Запустите следующую команду для проверки экспорта:

```bash
python main.py --source input/video.mp4 --output output/test_result.mp4 --export-data --classes person --no-display
```

## Что должно произойти

1. **Создание директории запуска:**
   ```
   output/run_YYYYMMDD_HHMMSS/
   ├── analytics/
   │   ├── detection_analysis_trajectories.json
   │   ├── detection_analysis_trajectories.csv
   │   ├── detection_analysis_frames.json
   │   ├── detection_analysis_stats.json
   │   ├── detection_analysis_uwb_format.csv
   │   └── summary.txt
   ├── test_result.mp4
   └── floor_projection.mp4
   ```

2. **Вывод в консоли:**
   ```
   2025-11-23 22:30:15 - INFO - Экспорт данных: ВКЛЮЧЕН (в output/run_20251123_223015/analytics)
   ...
   2025-11-23 22:31:45 - INFO - 
   Экспорт данных трекинга...
   
   2025-11-23 22:31:46 - INFO - 
   Экспортированные файлы:
     trajectories_json: output/run_20251123_223015/analytics/detection_analysis_trajectories.json
     trajectories_csv: output/run_20251123_223015/analytics/detection_analysis_trajectories.csv
     ...
   ```

## Проверка содержимого файлов

### 1. Проверка trajectories.json
```bash
# Windows PowerShell
Get-Content output/run_*/analytics/detection_analysis_trajectories.json | Select-Object -First 20

# Linux/Mac
head -20 output/run_*/analytics/detection_analysis_trajectories.json
```

Должен содержать JSON с полями:
- `export_date`
- `total_tracks`
- `trajectories` (массив с track_id, class, points, metadata)

### 2. Проверка trajectories.csv
```bash
# Windows PowerShell
Get-Content output/run_*/analytics/detection_analysis_trajectories.csv | Select-Object -First 10

# Linux/Mac
head -10 output/run_*/analytics/detection_analysis_trajectories.csv
```

Должен содержать CSV с заголовками:
```
track_id,class,x,y,timestamp,proj_x,proj_y
```

### 3. Проверка summary.txt
```bash
# Windows PowerShell
Get-Content output/run_*/analytics/summary.txt

# Linux/Mac
cat output/run_*/analytics/summary.txt
```

Должен содержать текстовый отчет с:
- Датой экспорта
- Количеством кадров
- Количеством треков
- Статистикой по классам

## Расширенные тесты

### Тест с адаптивным трекингом
```bash
python main.py --source input/video.mp4 --export-data --adaptive-tracking --classes person --no-display
```

В файлах должно появиться поле `final_state` со значениями: STATIC, DYNAMIC, MOVABLE

### Тест с распознаванием действий
```bash
python main.py --source input/video.mp4 --export-data --action-recognition --pose-detection --classes person --no-display
```

В `detection_analysis_frames.json` должно появиться поле `action` с:
- `label`: standing, sitting, walking, running, falling
- `confidence`: 0.0-1.0

### Тест со всеми объектами
```bash
python main.py --source input/video.mp4 --export-data --classes all --no-display
```

В файлах должны быть треки разных классов (chair, person, etc.)

## Устранение проблем

### Проблема: Папка analytics не создается
**Решение:** Убедитесь, что флаг `--export-data` указан в команде запуска.

### Проблема: Файлы пустые или содержат только заголовки
**Решение:** 
1. Проверьте, что в видео есть объекты указанных классов
2. Попробуйте снизить `--conf` (порог уверенности): `--conf 0.3`
3. Увеличьте длительность видео или используйте другое видео

### Проблема: Ошибка при экспорте
**Решение:**
1. Проверьте права на запись в папку `output/`
2. Убедитесь, что установлены все зависимости: `pip install -r requirements.txt`
3. Проверьте логи на наличие трассировки ошибки

## Проверка целостности данных

### Python скрипт для проверки
```python
import json
from pathlib import Path

# Найти последний запуск
runs = sorted(Path('output').glob('run_*'))
if not runs:
    print("Нет запусков для проверки")
    exit(1)

latest_run = runs[-1]
analytics_dir = latest_run / 'analytics'

# Проверка наличия файлов
required_files = [
    'detection_analysis_trajectories.json',
    'detection_analysis_trajectories.csv',
    'summary.txt'
]

for filename in required_files:
    filepath = analytics_dir / filename
    if filepath.exists():
        print(f"✓ {filename} существует ({filepath.stat().st_size} байт)")
    else:
        print(f"✗ {filename} отсутствует!")

# Проверка содержимого JSON
json_file = analytics_dir / 'detection_analysis_trajectories.json'
if json_file.exists():
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"\n✓ JSON валиден")
        print(f"  Всего треков: {data.get('total_tracks', 0)}")
        if data.get('trajectories'):
            print(f"  Первый трек: ID={data['trajectories'][0]['track_id']}, "
                  f"Класс={data['trajectories'][0]['class']}, "
                  f"Точек={len(data['trajectories'][0]['points'])}")
```

Сохраните как `check_export.py` и запустите:
```bash
python check_export.py
```

## Ожидаемый результат успешного теста

```
✓ detection_analysis_trajectories.json существует (45231 байт)
✓ detection_analysis_trajectories.csv существует (12456 байт)
✓ summary.txt существует (892 байт)

✓ JSON валиден
  Всего треков: 5
  Первый трек: ID=1, Класс=person, Точек=342
```


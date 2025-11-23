# 📋 Шпаргалка команд

## Базовые команды

### Простая детекция
```bash
python main.py --source input/video.mp4
```

### Только динамические объекты (НОВОЕ!)
```bash
python main.py --source input/video.mp4 --dynamic-only
```

### С настройкой порога движения
```bash
python main.py --source input/video.mp4 --dynamic-only --motion-threshold 3.0
```

### С экспортом данных (ИСПРАВЛЕНО!)
```bash
python main.py --source input/video.mp4 --export-data
```

### Без GUI (для сервера)
```bash
python main.py --source input/video.mp4 --no-display --export-data
```

---

## Трекинг

### Стандартный трекинг
```bash
python main.py --source input/video.mp4 --enable-tracking
```

### Адаптивный трекинг (умный)
```bash
python main.py --source input/video.mp4 --adaptive-tracking
```

### Без трекинга
```bash
python main.py --source input/video.mp4 --no-tracking
```

---

## Визуализация

### С траекториями
```bash
python main.py --source input/video.mp4 --show-trajectories
```

### С heat map
```bash
python main.py --source input/video.mp4 --show-heatmap
```

### С панелью статистики
```bash
python main.py --source input/video.mp4 --show-stats-panel
```

### Всё вместе
```bash
python main.py --source input/video.mp4 --show-trajectories --show-heatmap --show-stats-panel
```

---

## Распознавание действий

### Только позы (скелеты)
```bash
python main.py --source input/video.mp4 --pose-detection
```

### С распознаванием действий
```bash
python main.py --source input/video.mp4 --action-recognition --pose-detection
```

---

## Классы объектов

### Только люди
```bash
python main.py --source input/video.mp4 --classes person
```

### Люди и мебель
```bash
python main.py --source input/video.mp4 --classes "person,chair,couch"
```

### Все объекты
```bash
python main.py --source input/video.mp4 --classes all
```

### Список доступных классов
```bash
python main.py --list-classes
```

---

## Модели

### Быстрая (Nano)
```bash
python main.py --source input/video.mp4 --model yolov8n
```

### Точная (X-Large)
```bash
python main.py --source input/video.mp4 --model yolov8x
```

---

## Параметры детекции

### Низкий порог уверенности (больше детекций)
```bash
python main.py --source input/video.mp4 --conf 0.3
```

### Высокий порог (меньше ложных срабатываний)
```bash
python main.py --source input/video.mp4 --conf 0.7
```

### Высокое разрешение инференса (лучше для мелких объектов)
```bash
python main.py --source input/video.mp4 --imgsz 1280
```

---

## Полный набор функций

### Максимальная конфигурация
```bash
python main.py \
  --source input/video.mp4 \
  --output output/result.mp4 \
  --model yolov8x \
  --classes person \
  --conf 0.5 \
  --imgsz 1280 \
  --adaptive-tracking \
  --action-recognition \
  --pose-detection \
  --show-trajectories \
  --show-heatmap \
  --show-stats-panel \
  --export-data
```

### Режим для сервера (без GUI)
```bash
python main.py \
  --source input/video.mp4 \
  --output output/result.mp4 \
  --no-display \
  --adaptive-tracking \
  --export-data \
  --classes person
```

---

## Горячие клавиши (во время работы)

| Клавиша | Действие |
|---------|----------|
| `q` | Выход |
| `p` | Показать/скрыть скелеты |
| `a` | Включить/выключить распознавание действий |
| `h` | Показать/скрыть heat map |
| `t` | Показать/скрыть траектории |
| `s` | Показать/скрыть панель статистики |

---

## Проверка экспорта

### Быстрый тест
```bash
python main.py --source input/video.mp4 --export-data --classes person --no-display
```

### Проверка результатов
```bash
# Windows
dir output\run_*\analytics\

# Linux/Mac
ls -la output/run_*/analytics/
```

### Python скрипт проверки
```python
from pathlib import Path
import json

runs = sorted(Path('output').glob('run_*'))
latest = runs[-1] / 'analytics'

for file in ['detection_analysis_trajectories.json', 
             'detection_analysis_trajectories.csv',
             'summary.txt']:
    path = latest / file
    print(f"{'✓' if path.exists() else '✗'} {file}")
```

---

## Устранение проблем

### Экспорт не работает
```bash
# Убедитесь, что флаг --export-data указан
python main.py --source input/video.mp4 --export-data

# Проверьте права на запись
ls -la output/
```

### Низкая производительность
```bash
# Используйте меньшую модель
python main.py --source input/video.mp4 --model yolov8n

# Уменьшите разрешение инференса
python main.py --source input/video.mp4 --imgsz 640

# Отключите лишние функции
python main.py --source input/video.mp4 --no-tracking
```

### Мало детекций
```bash
# Снизьте порог уверенности
python main.py --source input/video.mp4 --conf 0.3

# Увеличьте разрешение инференса
python main.py --source input/video.mp4 --imgsz 1280
```

---

## Полезные ссылки

- **QUICK_START.md** - быстрый старт
- **BUGFIX_EXPORT.md** - исправление бага экспорта
- **TEST_EXPORT.md** - тестирование экспорта
- **ROADMAP.md** - планы развития


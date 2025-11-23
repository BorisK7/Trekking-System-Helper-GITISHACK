
"""Детектирование объектов на видео с использованием YOLOv8

Автор: YOLOv8 Object Detection System
Дата: 2025
Описание: Проект для детектирования объектов на видео с RTX 3060
"""

import argparse
import logging
import sys
from pathlib import Path

from utils import VideoProcessor, Config


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Детектирование объектов на видео с YOLOv8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Обработка видеофайла (все объекты)
  python main.py --source input/video.mp4 --output output/result.mp4

  # Детекция мебели, вещей и людей
  python main.py --source input/video.mp4 --classes "person,chair,couch,bed,dining table,potted plant,vase"

  # Детекция только стульев
  python main.py --source input/video.mp4 --classes chair

  # Детекция электроники
  python main.py --source input/video.mp4 --classes "tv,laptop,keyboard,mouse,cell phone"

  # Детекция предметов интерьера
  python main.py --source input/video.mp4 --classes "chair,couch,potted plant,vase,clock,book"

  # Использование веб-камеры
  python main.py --source 0 --output output/webcam.mp4

  # Без отображения (только сохранение)
  python main.py --source input/video.mp4 --output output/result.mp4 --no-display

  # С трекингом и траекториями
  python main.py --source input/video.mp4 --output output/tracked.mp4 --show-trajectories

  # С heat map и панелью статистики
  python main.py --source input/video.mp4 --show-heatmap --show-stats-panel

  # С экспортом данных для сравнения с UWB
  python main.py --source input/video.mp4 --export-data --classes person

  # Без трекинга (только детекция)
  python main.py --source input/video.mp4 --no-tracking

Доступные модели:
  - yolov8n: Nano (самая быстрая)
  - yolov8s: Small
  - yolov8m: Medium (по умолчанию)
  - yolov8l: Large
  - yolov8x: Extra Large (самая точная)

Примеры классов COCO:
  Мебель и интерьер: chair, couch, bed, dining table, potted plant, vase, clock
  Электроника: tv, laptop, cell phone, keyboard, mouse, remote
  Предметы: book, bottle, cup, bowl, scissors, teddy bear, suitcase, backpack
  Кухня: microwave, oven, toaster, sink, refrigerator, fork, knife, spoon
  Люди и животные: person, dog, cat, bird, horse, cow
  Транспорт: car, truck, bus, motorcycle, bicycle
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8x',
        choices=list(Config.AVAILABLE_MODELS.keys()),
        help='Модель YOLOv8 для использования (по умолчанию: yolov8x)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Путь к видеофайлу или 0 для веб-камеры'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Путь для сохранения результата (опционально)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=Config.DEFAULT_CONF_THRESHOLD,
        help=f'Порог уверенности детекции (0-1, по умолчанию: {Config.DEFAULT_CONF_THRESHOLD})'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=Config.DEFAULT_IOU_THRESHOLD,
        help=f'Порог IoU для NMS (0-1, по умолчанию: {Config.DEFAULT_IOU_THRESHOLD})'
    )
    
    parser.add_argument(
        '--classes',
        type=str,
        default='all',
        help='Классы для детекции: "all" (все), "person" (люди), "person,car" (несколько) или ID классов "0,2,3"'
    )
    
    parser.add_argument(
        '--list-classes',
        action='store_true',
        help='Показать список всех доступных классов и выйти'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Не отображать результат в реальном времени'
    )
    
    parser.add_argument(
        '--enable-tracking',
        action='store_true',
        default=True,
        help='Включить трекинг объектов (по умолчанию)'
    )
    
    parser.add_argument(
        '--no-tracking',
        action='store_true',
        help='Отключить трекинг объектов'
    )
    
    parser.add_argument(
        '--show-trajectories',
        action='store_true',
        default=True,
        help='Показывать траектории движения (по умолчанию при трекинге)'
    )
    
    parser.add_argument(
        '--show-heatmap',
        action='store_true',
        help='Показывать heat map активности'
    )
    
    parser.add_argument(
        '--show-stats-panel',
        action='store_true',
        help='Показывать панель статистики справа'
    )
    
    parser.add_argument(
        '--export-data',
        action='store_true',
        help='Экспортировать данные трекинга в JSON/CSV'
    )
    
    parser.add_argument(
        '--export-dir',
        type=str,
        default='output/analytics',
        help='Директория для экспорта данных (по умолчанию: output/analytics)'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Валидация аргументов"""
    # Проверка порога уверенности
    if not 0 <= args.conf <= 1:
        logger.error("Порог уверенности должен быть в диапазоне [0, 1]")
        return False
    
    # Проверка порога IoU
    if not 0 <= args.iou <= 1:
        logger.error("Порог IoU должен быть в диапазоне [0, 1]")
        return False
    
    # Проверка источника видео
    if args.source not in ['0', 0]:
        source_path = Path(args.source)
        if not source_path.exists():
            logger.error(f"Видеофайл не найден: {args.source}")
            return False
    
    # Если нет вывода и нет отображения - предупреждение
    if args.no_display and args.output is None:
        logger.warning("Внимание: результат не будет отображаться и не будет сохранен!")
        response = input("Продолжить? (y/n): ")
        if response.lower() != 'y':
            return False
    
    return True


def main():
    """Главная функция"""
    try:
        # Вывод заголовка
        print("=" * 70)
        print("YOLOv8 OBJECT DETECTION SYSTEM")
        print("=" * 70)
        print()
        
        # Парсинг аргументов
        args = parse_arguments()
        
        # Вывод списка классов, если запрошено
        if args.list_classes:
            print("\nДоступные классы COCO (80 классов):\n")
            for class_id, class_name in sorted(Config.COCO_CLASSES.items()):
                print(f"  {class_id:2d}: {class_name}")
            print("\nИспользование:")
            print("  --classes person          # Только люди")
            print("  --classes \"person,car\"    # Люди и машины")
            print("  --classes \"0,2,3\"         # По ID (person, car, motorcycle)")
            print("  --classes all             # Все классы (по умолчанию)")
            sys.exit(0)
        
        # Валидация
        if not validate_arguments(args):
            sys.exit(1)
        
        # Парсинг классов
        try:
            classes = Config.parse_classes(args.classes)
            if classes:
                class_names = [Config.COCO_CLASSES[cid] for cid in classes]
                classes_str = ', '.join(class_names)
            else:
                classes_str = 'Все классы (80)'
        except ValueError as e:
            logger.error(f"Ошибка в параметре --classes: {e}")
            sys.exit(1)
        
        # Вывод конфигурации
        logger.info("КОНФИГУРАЦИЯ:")
        logger.info(f"  Модель: {args.model}")
        logger.info(f"  Источник: {args.source}")
        logger.info(f"  Выходной файл: {args.output or 'Не задан'}")
        logger.info(f"  Классы: {classes_str}")
        logger.info(f"  Порог уверенности: {args.conf}")
        logger.info(f"  Порог IoU: {args.iou}")
        logger.info(f"  Отображение: {'Нет' if args.no_display else 'Да'}")
        logger.info(f"  Устройство: {Config.get_device_info()}")
        print()
        
        # Определение параметров
        enable_tracking = args.enable_tracking and not args.no_tracking
        enable_analytics = enable_tracking  # Аналитика требует трекинга
        
        # Вывод расширенных параметров
        if enable_tracking:
            logger.info(f"  Трекинг: ВКЛ")
            logger.info(f"  Траектории: {'ВКЛ' if args.show_trajectories else 'ВЫКЛ'}")
            logger.info(f"  Heat map: {'ВКЛ' if args.show_heatmap else 'ВЫКЛ'}")
            logger.info(f"  Панель статистики: {'ВКЛ' if args.show_stats_panel else 'ВЫКЛ'}")
        else:
            logger.info(f"  Трекинг: ВЫКЛ")
        
        if args.export_data:
            logger.info(f"  Экспорт данных: ВКЛ ({args.export_dir})")
        
        print()
        
        # Создание процессора
        processor = VideoProcessor(
            model_name=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            classes=classes,
            enable_tracking=enable_tracking,
            enable_analytics=enable_analytics,
            enable_export=args.export_data
        )
        
        # Обработка видео
        processor.process_video(
            source=args.source,
            output_path=args.output,
            display=not args.no_display,
            show_trajectories=args.show_trajectories and enable_tracking,
            show_heatmap=args.show_heatmap and enable_analytics,
            show_stats_panel=args.show_stats_panel and enable_tracking,
            create_floor_projection=enable_tracking,  # Создаем проекцию если трекинг включен
            projection_output='output/floor_projection.mp4' if args.output else None
        )
        
        logger.info("Обработка завершена успешно!")
        
    except KeyboardInterrupt:
        logger.info("\nПрервано пользователем")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

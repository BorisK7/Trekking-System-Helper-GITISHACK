"""
Класс для обработки видео с детектированием людей
"""

import cv2
import time
import logging
from pathlib import Path
from ultralytics import YOLO
from .config import Config


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Обработчик видео для детектирования объектов"""
    
    def __init__(self, model_name='yolov8m', conf_threshold=0.5, iou_threshold=0.45, classes=None):
        """
        Инициализация процессора
        
        Args:
            model_name: Название модели YOLOv8
            conf_threshold: Порог уверенности детекции (0-1)
            iou_threshold: Порог IoU для NMS
            classes: Список ID классов для детекции (None = все классы)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        
        # Загрузка модели
        model_path = Config.validate_model(model_name)
        logger.info(f"Загрузка модели {model_name}...")
        self.model = YOLO(model_path)
        
        # Перенос модели на GPU
        self.device = Config.DEVICE
        logger.info(f"Устройство: {Config.get_device_info()}")
        
        # Статистика
        self.frame_count = 0
        self.total_detections = 0
        self.class_counts = {}  # Подсчет по классам
        
    def process_video(self, source, output_path=None, display=True):
        """
        Обработка видео с детектированием людей
        
        Args:
            source: Путь к видеофайлу или 0 для веб-камеры
            output_path: Путь для сохранения результата (опционально)
            display: Отображать результат в реальном времени
        """
        # Открытие источника видео
        cap = self._open_video_source(source)
        if cap is None:
            return
        
        # Получение параметров видео
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(source, str) else 0
        
        logger.info(f"Разрешение: {width}x{height}, FPS: {fps}")
        if total_frames > 0:
            logger.info(f"Всего кадров: {total_frames}")
        
        # Настройка записи результата
        writer = None
        if output_path:
            writer = self._setup_video_writer(output_path, fps, width, height)
        
        # Переменные для FPS
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        logger.info("Начало обработки видео...")
        logger.info("Нажмите 'q' для выхода, 'p' для паузы/продолжения")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        logger.info("Достигнут конец видео")
                        break
                    
                    # Детектирование объектов
                    processed_frame, detections = self._detect_objects(frame)
                    
                    # Обновление статистики
                    self.frame_count += 1
                    self.total_detections += len(detections)
                    
                    # Подсчет по классам
                    for det in detections:
                        class_id = det[4]
                        class_name = Config.COCO_CLASSES.get(class_id, 'unknown')
                        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
                    fps_frame_count += 1
                    
                    # Расчет FPS
                    elapsed_time = time.time() - fps_start_time
                    if elapsed_time > 1.0:
                        current_fps = fps_frame_count / elapsed_time
                        fps_frame_count = 0
                        fps_start_time = time.time()
                    
                    # Добавление информации на кадр
                    info_frame = self._add_info_overlay(
                        processed_frame, 
                        len(detections), 
                        current_fps,
                        self.frame_count,
                        total_frames
                    )
                    
                    # Сохранение результата
                    if writer:
                        writer.write(info_frame)
                    
                    # Отображение результата
                    if display:
                        cv2.imshow('YOLOv8 People Detection', info_frame)
                
                # Обработка нажатий клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Остановка по запросу пользователя")
                    break
                elif key == ord('p'):
                    paused = not paused
                    status = "ПАУЗА" if paused else "ПРОДОЛЖЕНИЕ"
                    logger.info(f"Статус: {status}")
                
                # Отображение прогресса
                if self.frame_count % 100 == 0 and total_frames > 0:
                    progress = (self.frame_count / total_frames) * 100
                    logger.info(f"Обработано: {self.frame_count}/{total_frames} кадров ({progress:.1f}%)")
        
        except KeyboardInterrupt:
            logger.info("Прервано пользователем")
        
        finally:
            # Освобождение ресурсов
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            # Вывод итоговой статистики
            self._print_statistics(total_frames)
    
    def _open_video_source(self, source):
        """Открытие источника видео"""
        try:
            if source == 0 or source == '0':
                logger.info("Открытие веб-камеры...")
                cap = cv2.VideoCapture(0)
            else:
                source_path = Path(source)
                if not source_path.exists():
                    logger.error(f"Файл не найден: {source}")
                    return None
                logger.info(f"Открытие видеофайла: {source}")
                cap = cv2.VideoCapture(str(source_path))
            
            if not cap.isOpened():
                logger.error("Не удалось открыть источник видео")
                return None
            
            return cap
        
        except Exception as e:
            logger.error(f"Ошибка при открытии источника: {e}")
            return None
    
    def _setup_video_writer(self, output_path, fps, width, height):
        """Настройка записи видео"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )
            
            if not writer.isOpened():
                logger.warning("Не удалось создать файл для записи")
                return None
            
            logger.info(f"Результат будет сохранен в: {output_path}")
            return writer
        
        except Exception as e:
            logger.error(f"Ошибка при настройке записи: {e}")
            return None
    
    def _detect_objects(self, frame):
        """
        Детектирование объектов на кадре
        
        Returns:
            processed_frame: Кадр с нарисованными боксами
            detections: Список детекций [(x1, y1, x2, y2, class_id, conf), ...]
        """
        # Запуск детекции
        results = self.model(
            frame,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,  # Фильтр по классам (None = все)
            verbose=False
        )
        
        detections = []
        processed_frame = frame.copy()
        
        # Обработка результатов
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                # Получение координат, класса и уверенности
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = Config.COCO_CLASSES.get(class_id, 'unknown')
                
                detections.append((x1, y1, x2, y2, class_id, conf))
                
                # Получение цвета для класса
                box_color = Config.get_category_color(class_id)
                
                # Рисование бокса
                cv2.rectangle(
                    processed_frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    Config.BOX_THICKNESS
                )
                
                # Подготовка текста
                label = f"{class_name.title()} {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    Config.FONT,
                    Config.FONT_SCALE,
                    Config.FONT_THICKNESS
                )
                
                # Фон для текста (затемненный цвет бокса)
                text_bg_color = tuple(int(c * 0.7) for c in box_color)
                cv2.rectangle(
                    processed_frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    text_bg_color,
                    -1
                )
                
                # Текст
                cv2.putText(
                    processed_frame,
                    label,
                    (x1, y1 - baseline - 2),
                    Config.FONT,
                    Config.FONT_SCALE,
                    Config.TEXT_COLOR,
                    Config.FONT_THICKNESS
                )
        
        return processed_frame, detections
    
    def _add_info_overlay(self, frame, num_objects, fps, current_frame, total_frames):
        """Добавление информационного оверлея на кадр"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Полупрозрачная панель сверху
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Информация
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Objects: {num_objects}",
        ]
        
        if total_frames > 0:
            info_lines.append(f"Frame: {current_frame}/{total_frames}")
        else:
            info_lines.append(f"Frame: {current_frame}")
        
        # Вывод информации
        y_offset = 25
        for line in info_lines:
            cv2.putText(
                frame,
                line,
                (10, y_offset),
                Config.FONT,
                0.7,
                (255, 255, 255),
                2
            )
            y_offset += 25
        
        return frame
    
    def _print_statistics(self, total_frames):
        """Вывод итоговой статистики"""
        logger.info("=" * 50)
        logger.info("СТАТИСТИКА ОБРАБОТКИ")
        logger.info("=" * 50)
        logger.info(f"Обработано кадров: {self.frame_count}")
        if total_frames > 0:
            logger.info(f"Общее количество кадров: {total_frames}")
        logger.info(f"Всего обнаружено объектов: {self.total_detections}")
        if self.frame_count > 0:
            avg_objects = self.total_detections / self.frame_count
            logger.info(f"Среднее количество объектов на кадр: {avg_objects:.2f}")
        
        # Статистика по классам
        if self.class_counts:
            logger.info("\nРаспределение по классам:")
            sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes[:10]:  # Топ-10
                logger.info(f"  {class_name}: {count}")
            if len(sorted_classes) > 10:
                logger.info(f"  ... и еще {len(sorted_classes) - 10} классов")
        
        logger.info("=" * 50)

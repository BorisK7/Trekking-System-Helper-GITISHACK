"""
Расширенный класс для обработки видео с трекингом, аналитикой и визуализацией
"""

import cv2
import time
import logging
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from .config import Config
from .tracker import ObjectTracker
from .analytics import SceneAnalytics, OccupancyAnalyzer
from .visualizer import TrajectoryVisualizer, HeatMapVisualizer, ZoneVisualizer, StatisticsOverlay
from .distance import DistanceEstimator
from .exporter import DataExporter


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoProcessorAdvanced:
    """Расширенный обработчик видео с трекингом и аналитикой"""
    
    def __init__(self, model_name='yolov8m', conf_threshold=0.5, iou_threshold=0.45, classes=None,
                 enable_tracking=True, enable_analytics=True, enable_export=False):
        """
        Инициализация процессора
        
        Args:
            model_name: Название модели YOLOv8
            conf_threshold: Порог уверенности детекции (0-1)
            iou_threshold: Порог IoU для NMS
            classes: Список ID классов для детекции (None = все классы)
            enable_tracking: Включить трекинг объектов
            enable_analytics: Включить аналитику (heat map, зоны)
            enable_export: Включить экспорт данных
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.enable_tracking = enable_tracking
        self.enable_analytics = enable_analytics
        self.enable_export = enable_export
        
        # Загрузка модели
        model_path = Config.validate_model(model_name)
        logger.info(f"Загрузка модели {model_name}...")
        self.model = YOLO(model_path)
        
        # Явный перенос модели на GPU
        self.device = Config.DEVICE
        if self.device == 'cuda':
            self.model.to('cuda')
            logger.info(f"Модель перенесена на GPU: {Config.get_device_info()}")
        else:
            logger.info(f"Устройство: {Config.get_device_info()}")
        
        # Статистика
        self.frame_count = 0
        self.total_detections = 0
        self.class_counts = {}
        
        # Модули (инициализируются при обработке видео)
        self.tracker = None
        self.analytics = None
        self.occupancy = None
        self.distance_estimator = None
        self.exporter = None
        
        # Визуализаторы
        self.traj_visualizer = TrajectoryVisualizer()
        self.heatmap_visualizer = HeatMapVisualizer()
        self.zone_visualizer = ZoneVisualizer()
        self.stats_overlay = StatisticsOverlay()
    
    def process_video(self, source, output_path=None, display=True, 
                     show_trajectories=True, show_heatmap=False, show_stats_panel=False,
                     create_floor_projection=True, projection_output=None):
        """
        Обработка видео с детектированием и трекингом
        
        Args:
            source: Путь к видеофайлу или 0 для веб-камеры
            output_path: Путь для сохранения результата
            display: Отображать результат в реальном времени
            show_trajectories: Показывать траектории движения
            show_heatmap: Показывать heat map активности
            show_stats_panel: Показывать панель статистики
            create_floor_projection: Создавать проекцию на плоскость пола
            projection_output: Путь для сохранения проекции (по умолчанию output/floor_projection.mp4)
        """
        # Открытие видео
        cap = self._open_video_source(source)
        if cap is None:
            return
        
        # Получение параметров видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Параметры видео: {width}x{height} @ {fps}fps, {total_frames} кадров")
        
        # Инициализация модулей
        if self.enable_tracking:
            self.tracker = ObjectTracker(max_age=30, min_hits=3)
            logger.info("Трекинг объектов: ВКЛЮЧЕН")
        
        if self.enable_analytics:
            self.analytics = SceneAnalytics((height, width))
            self.occupancy = OccupancyAnalyzer()
            logger.info("Аналитика сцены: ВКЛЮЧЕНА")
        
        if self.enable_export:
            self.exporter = DataExporter()
            logger.info("Экспорт данных: ВКЛЮЧЕН")
        
        self.distance_estimator = DistanceEstimator(camera_height_m=2.0)
        
        # Настройка записи
        writer = None
        if output_path:
            writer = self._setup_video_writer(output_path, fps, width, height)
        
        # Настройка проекции на плоскость
        projection_writer = None
        projection_size = (800, 600)  # Размер проекции на плоскость
        if create_floor_projection:
            if projection_output is None:
                projection_output = 'output/floor_projection.mp4'
            projection_writer = self._setup_video_writer(projection_output, fps, projection_size[0], projection_size[1])
            logger.info(f"Проекция на плоскость: {projection_size[0]}x{projection_size[1]}")
        
        # Переменные для FPS
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        # Переменные для паузы
        paused = False
        
        logger.info("Начало обработки...")
        start_time = time.time()
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.info("Достигнут конец видео")
                        break
                    
                    # Временная метка
                    frame_time = self.frame_count / fps if fps > 0 else self.frame_count
                    
                    # Детектирование объектов
                    processed_frame, detections = self._detect_objects(frame)
                    
                    # Обновление статистики
                    self.frame_count += 1
                    self.total_detections += len(detections)
                    
                    # Трекинг
                    if self.enable_tracking and self.tracker:
                        tracked_objects = self.tracker.update(detections, frame_time)
                        
                        # Подсчет по классам
                        for track_id, x1, y1, x2, y2, class_id, conf in tracked_objects:
                            class_name = Config.COCO_CLASSES.get(class_id, 'unknown')
                            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
                    else:
                        # Без трекинга - используем детекции напрямую
                        tracked_objects = [(i, x1, y1, x2, y2, cid, conf) 
                                          for i, (x1, y1, x2, y2, cid, conf) in enumerate(detections)]
                    
                    fps_frame_count += 1
                    
                    # Расчет FPS
                    elapsed_time = time.time() - fps_start_time
                    if elapsed_time > 1.0:
                        current_fps = fps_frame_count / elapsed_time
                        fps_frame_count = 0
                        fps_start_time = time.time()
                    
                    # Аналитика
                    if self.enable_analytics and self.analytics:
                        self.analytics.update_heat_map(tracked_objects)
                        self.analytics.update_zones(tracked_objects)
                        self.analytics.update_timeline(frame_time, tracked_objects)
                    
                    # Визуализация
                    display_frame = processed_frame.copy()
                    
                    # Траектории
                    if show_trajectories and self.enable_tracking and self.tracker:
                        display_frame = self.traj_visualizer.draw_trajectories(display_frame, self.tracker)
                        display_frame = self.traj_visualizer.draw_track_info(
                            display_frame, tracked_objects, self.tracker
                        )
                    
                    # Heat map
                    if show_heatmap and self.enable_analytics and self.analytics:
                        heat_overlay, alpha = self.analytics.get_heat_map_overlay()
                        display_frame = self.heatmap_visualizer.overlay_heat_map(
                            display_frame, heat_overlay, alpha
                        )
                    
                    # Зоны (если определены)
                    if self.enable_analytics and self.analytics and self.analytics.zones:
                        display_frame = self.zone_visualizer.draw_zones(
                            display_frame, self.analytics.zones, self.analytics
                        )
                    
                    # Добавление информации на кадр
                    info_frame = self._add_info_overlay(
                        display_frame,
                        len(tracked_objects),
                        current_fps,
                        self.frame_count,
                        total_frames
                    )
                    
                    # Панель статистики
                    if show_stats_panel and self.enable_tracking:
                        info_frame = self.stats_overlay.draw_statistics_panel(
                            info_frame,
                            self.tracker,
                            self.analytics if self.enable_analytics else None,
                            self.occupancy if self.enable_analytics else None
                        )
                    
                    # Экспорт данных
                    if self.enable_export and self.exporter:
                        self.exporter.add_frame_data(
                            self.frame_count,
                            frame_time,
                            tracked_objects,
                            self.analytics
                        )
                    
                    # Создание проекции на плоскость
                    if create_floor_projection and projection_writer:
                        floor_frame = self._create_floor_projection(
                            tracked_objects, projection_size, frame_time
                        )
                        projection_writer.write(floor_frame)
                    
                    # Сохранение результата
                    if writer:
                        # Если есть панель статистики, нужно изменить размер
                        if show_stats_panel:
                            save_frame = cv2.resize(info_frame, (width, height))
                        else:
                            save_frame = info_frame
                        writer.write(save_frame)
                    
                    # Отображение результата
                    if display:
                        cv2.imshow('YOLOv8 Advanced Detection & Tracking', info_frame)
                
                # Обработка нажатий клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Остановка по запросу пользователя")
                    break
                elif key == ord('p'):
                    paused = not paused
                    status = "ПАУЗА" if paused else "ПРОДОЛЖЕНИЕ"
                    logger.info(f"Статус: {status}")
                elif key == ord('h'):
                    show_heatmap = not show_heatmap
                    logger.info(f"Heat map: {'ВКЛ' if show_heatmap else 'ВЫКЛ'}")
                elif key == ord('t'):
                    show_trajectories = not show_trajectories
                    logger.info(f"Траектории: {'ВКЛ' if show_trajectories else 'ВЫКЛ'}")
                elif key == ord('s'):
                    show_stats_panel = not show_stats_panel
                    logger.info(f"Панель статистики: {'ВКЛ' if show_stats_panel else 'ВЫКЛ'}")
                
                # Отображение прогресса
                if self.frame_count % 100 == 0 and total_frames > 0:
                    progress = (self.frame_count / total_frames) * 100
                    logger.info(f"Обработано: {self.frame_count}/{total_frames} кадров ({progress:.1f}%)")
        
        finally:
            # Освобождение ресурсов
            cap.release()
            if writer:
                writer.release()
            if projection_writer:
                projection_writer.release()
            if display:
                cv2.destroyAllWindows()
            
            # Статистика
            total_time = time.time() - start_time
            self._print_statistics(total_frames, total_time)
            
            # Экспорт траекторий
            if self.enable_export and self.exporter and self.enable_tracking and self.tracker:
                self._export_tracking_data()
    
    def _detect_objects(self, frame):
        """Детектирование объектов на кадре"""
        results = self.model(
            frame,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False
        )
        
        detections = []
        processed_frame = frame.copy()
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = Config.COCO_CLASSES.get(class_id, 'unknown')
                
                detections.append((x1, y1, x2, y2, class_id, conf))
                
                # Цвет для класса
                box_color = Config.get_category_color(class_id)
                
                # Рисование бокса
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_color, Config.BOX_THICKNESS)
                
                # Текст
                label = f"{class_name.title()} {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, Config.FONT, Config.FONT_SCALE, Config.FONT_THICKNESS
                )
                
                # Фон для текста
                text_bg_color = tuple(int(c * 0.7) for c in box_color)
                cv2.rectangle(
                    processed_frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    text_bg_color,
                    -1
                )
                
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
    
    def _open_video_source(self, source):
        """Открытие источника видео"""
        try:
            # Веб-камера
            if str(source) == '0' or source == 0:
                logger.info("Открытие веб-камеры...")
                cap = cv2.VideoCapture(0)
            else:
                # Файл
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
    
    def _print_statistics(self, total_frames, total_time):
        """Вывод итоговой статистики"""
        logger.info("=" * 50)
        logger.info("СТАТИСТИКА ОБРАБОТКИ")
        logger.info("=" * 50)
        logger.info(f"Обработано кадров: {self.frame_count}")
        if total_frames > 0:
            logger.info(f"Общее количество кадров: {total_frames}")
        logger.info(f"Время обработки: {total_time:.2f} сек")
        logger.info(f"Средний FPS: {self.frame_count / total_time:.2f}")
        logger.info(f"Всего обнаружено объектов: {self.total_detections}")
        
        if self.frame_count > 0:
            avg_objects = self.total_detections / self.frame_count
            logger.info(f"Среднее количество объектов на кадр: {avg_objects:.2f}")
        
        # Статистика по классам
        if self.class_counts:
            logger.info("\nРаспределение по классам:")
            sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes[:10]:
                logger.info(f"  {class_name}: {count}")
            if len(sorted_classes) > 10:
                logger.info(f"  ... и еще {len(sorted_classes) - 10} классов")
        
        # Статистика трекинга
        if self.enable_tracking and self.tracker:
            track_stats = self.tracker.get_statistics()
            logger.info(f"\nТреков создано: {track_stats['total_tracks_created']}")
            logger.info(f"Активных треков: {track_stats['active_tracks']}")
        
        logger.info("=" * 50)
    
    def _create_floor_projection(self, tracked_objects, projection_size, frame_time):
        """
        Создание проекции треков на плоскость пола (вид сверху)
        
        Args:
            tracked_objects: [(track_id, x1, y1, x2, y2, class_id, conf), ...]
            projection_size: (width, height) размер проекции
            frame_time: Временная метка кадра
            
        Returns:
            Кадр с проекцией на плоскость
        """
        width, height = projection_size
        
        # Создание фона (темный)
        floor_frame = np.zeros((height, width, 3), dtype=np.uint8)
        floor_frame[:] = (30, 30, 30)
        
        # Сетка для ориентации
        grid_color = (60, 60, 60)
        grid_step = 50
        for i in range(0, width, grid_step):
            cv2.line(floor_frame, (i, 0), (i, height), grid_color, 1)
        for i in range(0, height, grid_step):
            cv2.line(floor_frame, (0, i), (width, i), grid_color, 1)
        
        # Центральные оси
        cv2.line(floor_frame, (width//2, 0), (width//2, height), (100, 100, 100), 2)
        cv2.line(floor_frame, (0, height//2), (width, height//2), (100, 100, 100), 2)
        
        if not self.enable_tracking or not self.tracker:
            return floor_frame
        
        # Отрисовка траекторий
        for track_id, track in self.tracker.tracks.items():
            if track['hits'] < self.tracker.min_hits:
                continue
            
            trajectory = track['trajectory']
            if len(trajectory) < 2:
                continue
            
            # Цвет для трека
            color = self.traj_visualizer.get_color_for_track(track_id)
            
            # Проекция траектории на плоскость (нормализация координат)
            points = []
            for x, y, t in trajectory:
                # Простая проекция: нижняя часть экрана = ближе к камере
                # Верх экрана = дальше от камеры
                # Преобразуем координаты видео в координаты проекции
                proj_x = int((x / 1920) * width)  # Предполагаем 1920 ширину видео
                proj_y = int((y / 1080) * height)  # Предполагаем 1080 высоту видео
                points.append((proj_x, proj_y))
            
            # Отрисовка линии траектории (толстая линия)
            for i in range(1, len(points)):
                alpha = 0.3 + (i / len(points)) * 0.7
                thickness_current = max(2, int(4 * alpha))
                cv2.line(floor_frame, points[i-1], points[i], color, thickness_current)
            
            # Стрелка на последней точке
            if len(points) >= 2:
                self._draw_arrow_projection(floor_frame, points[-2], points[-1], color, 4)
            
            # Текущая позиция (круг)
            if points:
                cv2.circle(floor_frame, points[-1], 8, color, -1)
                cv2.circle(floor_frame, points[-1], 10, (255, 255, 255), 2)
                
                # ID трека
                class_name = Config.COCO_CLASSES.get(track['class_id'], 'unknown')
                label = f"#{track_id}: {class_name}"
                cv2.putText(
                    floor_frame,
                    label,
                    (points[-1][0] + 15, points[-1][1]),
                    Config.FONT,
                    0.5,
                    color,
                    2
                )
        
        # Информация
        cv2.putText(
            floor_frame,
            "Floor Projection (Top View)",
            (10, 30),
            Config.FONT,
            0.8,
            (255, 255, 255),
            2
        )
        cv2.putText(
            floor_frame,
            f"Active Tracks: {self.tracker.active_tracks_count if self.tracker else 0}",
            (10, 60),
            Config.FONT,
            0.6,
            (200, 200, 200),
            1
        )
        
        return floor_frame
    
    def _draw_arrow_projection(self, frame, pt1, pt2, color, thickness):
        """Отрисовка стрелки на проекции"""
        import numpy as np
        
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 5:
            return
        
        dx /= length
        dy /= length
        
        arrow_len = min(15, length * 0.3)
        arrow_angle = np.pi / 6
        
        arrow_pt1 = (
            int(pt2[0] - arrow_len * (dx * np.cos(arrow_angle) + dy * np.sin(arrow_angle))),
            int(pt2[1] - arrow_len * (dy * np.cos(arrow_angle) - dx * np.sin(arrow_angle)))
        )
        arrow_pt2 = (
            int(pt2[0] - arrow_len * (dx * np.cos(arrow_angle) - dy * np.sin(arrow_angle))),
            int(pt2[1] - arrow_len * (dy * np.cos(arrow_angle) + dx * np.sin(arrow_angle)))
        )
        
        cv2.line(frame, pt2, arrow_pt1, color, thickness)
        cv2.line(frame, pt2, arrow_pt2, color, thickness)
    
    def _export_tracking_data(self):
        """Экспорт данных трекинга"""
        logger.info("\nЭкспорт данных трекинга...")
        
        # Экспорт траекторий из трекера
        for track_id, track in self.tracker.tracks.items():
            if track['hits'] >= self.tracker.min_hits:
                class_name = Config.COCO_CLASSES.get(track['class_id'], 'unknown')
                
                metadata = {
                    'duration': self.tracker.get_track_duration(track_id),
                    'total_distance': self.tracker.get_total_distance(track_id),
                    'avg_speed': self.tracker.get_speed(track_id)
                }
                
                self.exporter.add_trajectory(
                    track_id,
                    track['trajectory'],
                    class_name,
                    metadata
                )
        
        # Статистика
        if self.enable_tracking:
            self.exporter.set_statistics(self.tracker.get_statistics())
        
        # Экспорт всех файлов
        exported = self.exporter.export_all()
        
        logger.info("\nЭкспортированные файлы:")
        for key, path in exported.items():
            logger.info(f"  {key}: {path}")
        
        # Текстовый отчет
        summary = self.exporter.create_summary_report()
        logger.info(f"  summary: {summary}")


# Обратная совместимость - алиас для базового процессора
VideoProcessor = VideoProcessorAdvanced

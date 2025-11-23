"""
Расширенный класс для обработки видео с трекингом, аналитикой и визуализацией
"""

import cv2
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from .config import Config
from .tracker import ObjectTracker
from .state_tracker import StateTracker # НОВЫЙ ТРЕКЕР
from .analytics import SceneAnalytics, OccupancyAnalyzer
from .visualizer import TrajectoryVisualizer, HeatMapVisualizer, ZoneVisualizer, StatisticsOverlay
from .distance import DistanceEstimator
from .exporter import DataExporter
from .action_recognition import ActionRecognizer
from .dynamic_filter import DynamicObjectFilter


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoProcessorAdvanced:
    """Расширенный обработчик видео с трекингом и аналитикой"""
    
    def __init__(self, model_name='yolov8m', conf_threshold=0.5, iou_threshold=0.45, classes=None,
                 enable_tracking=True, enable_analytics=True, enable_export=False,
                 enable_pose=False, enable_actions=False, use_adaptive_tracking=False,
                 imgsz=640, dynamic_only=False, motion_threshold=5.0): # Добавлены параметры
        """
        Инициализация процессора
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.enable_tracking = enable_tracking
        self.enable_analytics = enable_analytics
        self.enable_export = enable_export
        self.enable_pose = enable_pose
        self.enable_actions = enable_actions
        self.use_adaptive_tracking = use_adaptive_tracking
        self.imgsz = imgsz # Размер инференса
        self.dynamic_only = dynamic_only
        self.motion_threshold = motion_threshold
        
        if self.enable_actions:
            self.enable_pose = True
        
        self.model_det = None
        self.model_pose = None
        self.device = Config.DEVICE
        
        # 1. Загрузка модели ДЕТЕКЦИИ
        try:
            det_model_name = model_name.replace('-pose', '')
            logger.info(f"Загрузка модели детекции: {det_model_name}...")
            self.model_det = YOLO(det_model_name)
            if self.device == 'cuda':
                self.model_det.to('cuda')
        except Exception as e:
            logger.error(f"Ошибка загрузки модели детекции {det_model_name}: {e}")

        # 2. Загрузка модели ПОЗ
        if self.enable_pose:
            try:
                if model_name.endswith('.pt'):
                    pose_model_name = model_name.replace('.pt', '-pose.pt')
                else:
                    pose_model_name = f"{model_name}-pose"
                
                if '-pose' not in pose_model_name:
                     pose_model_name = pose_model_name.replace('.pt', '-pose.pt') if '.pt' in pose_model_name else f"{pose_model_name}-pose"

                logger.info(f"Загрузка модели поз: {pose_model_name}...")
                self.model_pose = YOLO(pose_model_name)
                if self.device == 'cuda':
                    self.model_pose.to('cuda')
            except Exception as e:
                logger.error(f"Ошибка загрузки модели поз {pose_model_name}: {e}")
                logger.warning("Отключение функционала поз и действий")
                self.enable_pose = False
                self.enable_actions = False

        if self.device == 'cuda':
             logger.info(f"Устройство: GPU {Config.get_device_info()}")
        else:
            logger.info(f"Устройство: {Config.get_device_info()}")
        
        self.frame_count = 0
        self.total_detections = 0
        self.class_counts = {}
        self.action_counts = {}
        
        self.video_width = 0
        self.video_height = 0
        self.video_fps = 0
        
        self.tracker = None
        self.analytics = None
        self.occupancy = None
        self.distance_estimator = None
        self.exporter = None
        self.action_recognizer = None
        self.dynamic_filter = None
        
        self.traj_visualizer = TrajectoryVisualizer()
        self.heatmap_visualizer = HeatMapVisualizer()
        self.zone_visualizer = ZoneVisualizer()
        self.stats_overlay = StatisticsOverlay()
    
    def process_video(self, source, output_path=None, display=True, 
                     show_trajectories=True, show_heatmap=False, show_stats_panel=False,
                     create_floor_projection=True, projection_output=None,
                     show_pose=True):
        
        cap = self._open_video_source(source)
        if cap is None: return
        
        self.video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Параметры видео: {self.video_width}x{self.video_height} @ {self.video_fps}fps, {total_frames} кадров")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path('output') / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Директория запуска: {run_dir}")
        
        if output_path:
            output_path = run_dir / Path(output_path).name
            
        if create_floor_projection:
            if projection_output:
                projection_output = run_dir / Path(projection_output).name
            else:
                projection_output = run_dir / 'floor_projection.mp4'
            logger.info(f"Проекция на плоскость: {projection_output}")

        if self.enable_tracking:
            if self.use_adaptive_tracking:
                self.tracker = StateTracker()
                logger.info("Трекинг объектов: АДАПТИВНЫЙ (StateTracker)")
            else:
                self.tracker = ObjectTracker(max_age=30, min_hits=3)
                logger.info("Трекинг объектов: СТАНДАРТНЫЙ (ObjectTracker)")
        
        if self.enable_analytics:
            self.analytics = SceneAnalytics((self.video_height, self.video_width))
            self.occupancy = OccupancyAnalyzer()
            logger.info("Аналитика сцены: ВКЛЮЧЕНА")
            
        if self.enable_actions:
            self.action_recognizer = ActionRecognizer()
            logger.info("Распознавание действий: ВКЛЮЧЕНО")
        
        if self.dynamic_only:
            self.dynamic_filter = DynamicObjectFilter(
                movement_threshold=self.motion_threshold,
                history_length=10,
                min_frames_to_confirm=3
            )
            logger.info(f"Фильтр динамических объектов: ВКЛЮЧЕН (порог: {self.motion_threshold} пикселей)")
        
        if self.enable_export:
            analytics_dir = run_dir / 'analytics'
            self.exporter = DataExporter(output_dir=analytics_dir)
            logger.info(f"Экспорт данных: ВКЛЮЧЕН (в {analytics_dir})")
        
        self.distance_estimator = DistanceEstimator(camera_height_m=2.0)
        
        writer = None
        if output_path:
            writer = self._setup_video_writer(output_path, self.video_fps, self.video_width, self.video_height)
        
        projection_writer = None
        projection_size = (1920, 1080) # Увеличиваем разрешение проекции
        if create_floor_projection:
            projection_writer = self._setup_video_writer(projection_output, self.video_fps, projection_size[0], projection_size[1])
            logger.info(f"Размер проекции: {projection_size[0]}x{projection_size[1]}")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        paused = False
        
        logger.info("Начало обработки...")
        start_time = time.time()
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_time = self.frame_count / self.video_fps if self.video_fps > 0 else self.frame_count
                    
                    processed_frame, detections, keypoints_data = self._detect_objects_hybrid(frame)
                    
                    self.frame_count += 1
                    self.total_detections += len(detections)
                    
                    current_actions = {}
                    track_keypoints = {}
                    
                    tracked_objects = []
                    motion_classifications = {}
                    
                    if self.enable_tracking and self.tracker:
                        tracked_objects = self.tracker.update(detections, frame_time)
                        
                        # Фильтрация динамических объектов
                        if self.dynamic_only and self.dynamic_filter:
                            tracked_objects, motion_classifications = self.dynamic_filter.update(
                                tracked_objects, frame_time
                            )
                        
                        if self.enable_pose and len(keypoints_data) > 0:
                            self._match_keypoints_to_tracks(tracked_objects, detections, keypoints_data, track_keypoints)
                            
                            if self.enable_actions and self.action_recognizer:
                                self._process_actions(tracked_objects, track_keypoints, current_actions)

                        # Подсчет (учитываем разный формат кортежей от трекеров)
                        for item in tracked_objects:
                             # Универсальное извлечение class_id (всегда 6-й элемент, индекс 5)
                            class_id = item[5]
                            class_name = Config.COCO_CLASSES.get(class_id, 'unknown')
                            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
                    else:
                        tracked_objects = [(i, x1, y1, x2, y2, cid, conf, 'UNKNOWN') 
                                          for i, (x1, y1, x2, y2, cid, conf) in enumerate(detections)]
                    
                    fps_frame_count += 1
                    elapsed_time = time.time() - fps_start_time
                    if elapsed_time > 1.0:
                        current_fps = fps_frame_count / elapsed_time
                        fps_frame_count = 0
                        fps_start_time = time.time()
                    
                    if self.enable_analytics and self.analytics:
                        self.analytics.update_heat_map(tracked_objects)
                        self.analytics.update_zones(tracked_objects)
                        self.analytics.update_timeline(frame_time, tracked_objects)
                        
                        people_tracks = []
                        furniture_tracks = []
                        furniture_classes = ['chair', 'couch', 'bed', 'bench', 'toilet']
                        
                        for trk in tracked_objects:
                            cid = trk[5]
                            cname = Config.COCO_CLASSES.get(cid, 'unknown')
                            if cname == 'person':
                                people_tracks.append(trk)
                            elif cname in furniture_classes:
                                furniture_tracks.append(trk)
                        
                        for ftrk in furniture_tracks:
                            fid = ftrk[0]
                            # Извлекаем координаты корректно
                            fx1, fy1, fx2, fy2 = ftrk[1:5]
                            fcid = ftrk[5]
                            fname = Config.COCO_CLASSES.get(fcid, 'unknown')
                            if fid not in self.occupancy.furniture_positions:
                                self.occupancy.register_furniture(fid, (fx1, fy1, fx2, fy2), fname)
                        
                        self.occupancy.update_occupancy(people_tracks, furniture_tracks)
                    
                    display_frame = processed_frame.copy()
                    
                    if self.enable_pose and show_pose:
                         display_frame = self.traj_visualizer.draw_skeletons(display_frame, track_keypoints)

                    if self.enable_tracking and self.tracker:
                        if show_trajectories:
                            display_frame = self.traj_visualizer.draw_trajectories(display_frame, self.tracker)
                        
                        display_frame = self.traj_visualizer.draw_track_info(
                            display_frame, tracked_objects, self.tracker, actions=current_actions, 
                            show_state=self.use_adaptive_tracking, motion_classifications=motion_classifications
                        )
                    
                    if show_heatmap and self.enable_analytics and self.analytics:
                        heat_overlay, alpha = self.analytics.get_heat_map_overlay()
                        display_frame = self.heatmap_visualizer.overlay_heat_map(display_frame, heat_overlay, alpha)
                    
                    if self.enable_analytics and self.analytics and self.analytics.zones:
                        display_frame = self.zone_visualizer.draw_zones(display_frame, self.analytics.zones, self.analytics)
                    
                    info_frame = self._add_info_overlay(display_frame, len(tracked_objects), current_fps, self.frame_count, total_frames)
                    
                    if show_stats_panel and self.enable_tracking:
                        info_frame = self.stats_overlay.draw_statistics_panel(
                            info_frame, self.tracker, self.analytics if self.enable_analytics else None,
                            self.occupancy if self.enable_analytics else None, action_stats=self.action_counts
                        )
                    
                    if self.enable_export and self.exporter:
                        self.exporter.add_frame_data(self.frame_count, frame_time, tracked_objects, self.analytics, actions=current_actions)
                    
                    if create_floor_projection and projection_writer:
                        floor_frame = self._create_floor_projection(tracked_objects, projection_size, frame_time)
                        projection_writer.write(floor_frame)
                    
                    if writer:
                        if show_stats_panel:
                            save_frame = cv2.resize(info_frame, (self.video_width, self.video_height))
                        else:
                            save_frame = info_frame
                        writer.write(save_frame)
                    
                    if display:
                        cv2.imshow('YOLOv8 Advanced Detection & Tracking', info_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('p'): show_pose = not show_pose
                elif key == ord('a'):
                    self.enable_actions = not self.enable_actions
                    if self.enable_actions and not self.action_recognizer:
                        self.action_recognizer = ActionRecognizer()
                elif key == ord('h'): show_heatmap = not show_heatmap
                elif key == ord('t'): show_trajectories = not show_trajectories
                elif key == ord('s'): show_stats_panel = not show_stats_panel
                
                if self.frame_count % 100 == 0 and total_frames > 0:
                    progress = (self.frame_count / total_frames) * 100
                    logger.info(f"Обработано: {self.frame_count}/{total_frames} кадров ({progress:.1f}%)")
        
        finally:
            cap.release()
            if writer: writer.release()
            if projection_writer: projection_writer.release()
            if display: cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            self._print_statistics(total_frames, total_time)
            
            if self.enable_export and self.exporter and self.enable_tracking and self.tracker:
                self._export_tracking_data()
    
    def _detect_objects_hybrid(self, frame):
        detections = []
        keypoints_data = []
        processed_frame = frame.copy()
        
        if self.model_det:
            results_det = self.model_det(
                frame,
                device=self.device,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                imgsz=self.imgsz, # Используем размер инференса
                verbose=False
            )
            
            if results_det and len(results_det) > 0:
                det_boxes = results_det[0].boxes
                for box in det_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = Config.COCO_CLASSES.get(class_id, 'unknown')
                    
                    detections.append((x1, y1, x2, y2, class_id, conf))
                    
                    box_color = Config.get_category_color(class_id)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_color, Config.BOX_THICKNESS)
                    label = f"{class_name.title()} {conf:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label, Config.FONT, Config.FONT_SCALE, Config.FONT_THICKNESS)
                    text_bg_color = tuple(int(c * 0.7) for c in box_color)
                    cv2.rectangle(processed_frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), text_bg_color, -1)
                    cv2.putText(processed_frame, label, (x1, y1 - baseline - 2), Config.FONT, Config.FONT_SCALE, Config.TEXT_COLOR, Config.FONT_THICKNESS)

        if self.enable_pose and self.model_pose:
            results_pose = self.model_pose(
                frame,
                device=self.device,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz, # Используем размер инференса
                verbose=False
            )
            
            if results_pose and len(results_pose) > 0:
                res = results_pose[0]
                if hasattr(res, 'keypoints') and res.keypoints is not None:
                    pose_boxes = res.boxes
                    raw_kpts = res.keypoints.data.cpu().numpy()
                    
                    aligned_keypoints = [None] * len(detections)
                    
                    for j, p_box in enumerate(pose_boxes):
                        px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                        p_box_coord = (px1, py1, px2, py2)
                        
                        best_iou = 0.5
                        best_det_idx = -1
                        
                        for i, det in enumerate(detections):
                            dx1, dy1, dx2, dy2, dcls, dconf = det
                            if dcls != 0: continue
                            
                            iou = self._calculate_iou(p_box_coord, (dx1, dy1, dx2, dy2))
                            if iou > best_iou:
                                best_iou = iou
                                best_det_idx = i
                        
                        if best_det_idx != -1:
                             aligned_keypoints[best_det_idx] = raw_kpts[j]
                    
                    keypoints_data = aligned_keypoints
                else:
                    keypoints_data = [None] * len(detections)
            else:
                keypoints_data = [None] * len(detections)
        else:
            keypoints_data = [None] * len(detections)
            
        return processed_frame, detections, keypoints_data
    
    def _match_keypoints_to_tracks(self, tracked_objects, detections, keypoints_data, track_keypoints):
        if len(keypoints_data) == 0: return
        
        used_indices = set()
        
        for item in tracked_objects:
            track_id = item[0]
            tx1, ty1, tx2, ty2 = item[1:5]
            
            best_iou = 0.5
            best_idx = -1
            track_box = (tx1, ty1, tx2, ty2)
            
            for i, (dx1, dy1, dx2, dy2, dcls, dconf) in enumerate(detections):
                if i in used_indices: continue
                
                det_box = (dx1, dy1, dx2, dy2)
                iou = self._calculate_iou(track_box, det_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_idx != -1:
                used_indices.add(best_idx)
                if best_idx < len(keypoints_data):
                    kpts = keypoints_data[best_idx]
                    if kpts is not None:
                         track_keypoints[track_id] = kpts
    
    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        if union_area == 0: return 0
        return inter_area / union_area

    def _process_actions(self, tracked_objects, track_keypoints, current_actions):
        track_boxes = {t[0]: t[1:5] for t in tracked_objects}
        active_ids = [t[0] for t in tracked_objects]
        self.action_recognizer.cleanup(active_ids)
        
        for track_id, kpts in track_keypoints.items():
            if track_id in track_boxes:
                bbox = track_boxes[track_id]
                vx, vy = 0, 0
                if track_id in self.tracker.tracks:
                     track = self.tracker.tracks[track_id]
                     if len(track['trajectory']) >= 2:
                         curr = track['trajectory'][-1]
                         prev = track['trajectory'][-2]
                         vx = curr[0] - prev[0]
                         vy = curr[1] - prev[1]
                
                action_label, conf = self.action_recognizer.update(track_id, kpts, bbox, (vx, vy))
                current_actions[track_id] = (action_label, conf)
                self.action_counts[action_label] = self.action_counts.get(action_label, 0) + 1

    def _open_video_source(self, source):
        try:
            if str(source) == '0' or source == 0:
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
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                logger.warning("Не удалось создать файл для записи")
                return None
            logger.info(f"Результат будет сохранен в: {output_path}")
            return writer
        except Exception as e:
            logger.error(f"Ошибка при настройке записи: {e}")
            return None

    def _add_info_overlay(self, frame, num_objects, fps, current_frame, total_frames):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        info_lines = [f"FPS: {fps:.1f}", f"Objects: {num_objects}"]
        if total_frames > 0:
            info_lines.append(f"Frame: {current_frame}/{total_frames}")
        else:
            info_lines.append(f"Frame: {current_frame}")
        y_offset = 25
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset), Config.FONT, 0.7, (255, 255, 255), 2)
            y_offset += 25
        return frame

    def _print_statistics(self, total_frames, total_time):
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
        if self.class_counts:
            logger.info("\nРаспределение по классам:")
            sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes[:10]:
                logger.info(f"  {class_name}: {count}")
        if self.action_counts:
            logger.info("\nСтатистика действий:")
            for action, count in self.action_counts.items():
                logger.info(f"  {action}: {count}")
        if self.enable_tracking and self.tracker:
            track_stats = self.tracker.get_statistics()
            logger.info(f"\nТреков создано: {track_stats['total_tracks_created']}")
            logger.info(f"Активных треков: {track_stats['active_tracks']}")
        logger.info("=" * 50)

    def _create_floor_projection(self, tracked_objects, projection_size, frame_time):
        width, height = projection_size
        floor_frame = np.zeros((height, width, 3), dtype=np.uint8)
        floor_frame[:] = (30, 30, 30)
        grid_color = (60, 60, 60)
        grid_step = 50
        for i in range(0, width, grid_step):
            cv2.line(floor_frame, (i, 0), (i, height), grid_color, 1)
        for i in range(0, height, grid_step):
            cv2.line(floor_frame, (0, i), (width, i), grid_color, 1)
        cv2.line(floor_frame, (width//2, 0), (width//2, height), (100, 100, 100), 2)
        cv2.line(floor_frame, (0, height//2), (width, height//2), (100, 100, 100), 2)
        
        if not self.enable_tracking or not self.tracker:
            return floor_frame
            
        for track_id, track in self.tracker.tracks.items():
            # Для StateTracker используем другую проверку (нет min_hits, но есть state)
            if self.use_adaptive_tracking:
                if track['state'] == 'UNKNOWN': continue
            else:
                 if track.get('hits', 0) < getattr(self.tracker, 'min_hits', 3):
                    continue

            trajectory = track['trajectory']
            if len(trajectory) < 2: continue
            
            color = self.traj_visualizer.get_color_for_track(track_id)
            # Если адаптивный трекинг, берем цвет состояния
            if self.use_adaptive_tracking:
                color = self.traj_visualizer.get_state_color(track['state'])

            points = []
            for x, y, t in trajectory:
                src_w = self.video_width if self.video_width > 0 else 1920
                src_h = self.video_height if self.video_height > 0 else 1080
                proj_x = int((x / src_w) * width)
                proj_y = int((y / src_h) * height)
                points.append((proj_x, proj_y))
            for i in range(1, len(points)):
                alpha = 0.3 + (i / len(points)) * 0.7
                thickness_current = max(2, int(4 * alpha))
                cv2.line(floor_frame, points[i-1], points[i], color, thickness_current)
            if len(points) >= 2:
                self._draw_arrow_projection(floor_frame, points[-2], points[-1], color, 4)
            if points:
                cv2.circle(floor_frame, points[-1], 8, color, -1)
                cv2.circle(floor_frame, points[-1], 10, (255, 255, 255), 2)
                class_name = Config.COCO_CLASSES.get(track['class_id'], 'unknown')
                label = f"#{track_id}: {class_name}"
                cv2.putText(floor_frame, label, (points[-1][0] + 15, points[-1][1]), Config.FONT, 0.5, color, 2)
        
        cv2.putText(floor_frame, "Floor Projection (Top View)", (10, 30), Config.FONT, 0.8, (255, 255, 255), 2)
        active_count = len(self.tracker.tracks) if self.tracker else 0
        cv2.putText(floor_frame, f"Active Tracks: {active_count}", (10, 60), Config.FONT, 0.6, (200, 200, 200), 1)
        return floor_frame

    def _draw_arrow_projection(self, frame, pt1, pt2, color, thickness):
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = np.sqrt(dx**2 + dy**2)
        if length < 5: return
        dx /= length
        dy /= length
        arrow_len = min(15, length * 0.3)
        arrow_angle = np.pi / 6
        arrow_pt1 = (int(pt2[0] - arrow_len * (dx * np.cos(arrow_angle) + dy * np.sin(arrow_angle))), int(pt2[1] - arrow_len * (dy * np.cos(arrow_angle) - dx * np.sin(arrow_angle))))
        arrow_pt2 = (int(pt2[0] - arrow_len * (dx * np.cos(arrow_angle) - dy * np.sin(arrow_angle))), int(pt2[1] - arrow_len * (dy * np.cos(arrow_angle) + dx * np.sin(arrow_angle))))
        cv2.line(frame, pt2, arrow_pt1, color, thickness)
        cv2.line(frame, pt2, arrow_pt2, color, thickness)

    def _export_tracking_data(self):
        logger.info("\nЭкспорт данных трекинга...")
        proj_width, proj_height = 1920, 1080 # ОБНОВЛЕНО для соответствия projection_size
        src_width = self.video_width if self.video_width > 0 else 1920
        src_height = self.video_height if self.video_height > 0 else 1080
        
        for track_id, track in self.tracker.tracks.items():
            # Фильтрация коротких треков
            if self.use_adaptive_tracking:
                if track['state'] == 'UNKNOWN' and track['age'] < 5: continue
            else:
                if track.get('hits', 0) < getattr(self.tracker, 'min_hits', 3): continue

            class_name = Config.COCO_CLASSES.get(track['class_id'], 'unknown')
            metadata = {
                'duration': self.tracker.get_track_duration(track_id),
                'total_distance': self.tracker.get_total_distance(track_id),
                'avg_speed': self.tracker.get_speed(track_id),
                'final_state': track.get('state', 'UNKNOWN') if self.use_adaptive_tracking else 'N/A'
            }
            extended_trajectory = []
            for x, y, t in track['trajectory']:
                proj_x = int((x / src_width) * proj_width)
                proj_y = int((y / src_height) * proj_height)
                state = track.get('state', 'N/A') # Можно было бы хранить историю состояний
                extended_trajectory.append((x, y, t, proj_x, proj_y))
            self.exporter.add_trajectory(track_id, extended_trajectory, class_name, metadata)
        
        if self.enable_tracking:
            self.exporter.set_statistics(self.tracker.get_statistics())
            
        exported = self.exporter.export_all()
        logger.info("\nЭкспортированные файлы:")
        for key, path in exported.items():
            logger.info(f"  {key}: {path}")
        summary = self.exporter.create_summary_report()
        logger.info(f"  summary: {summary}")

# Обратная совместимость
VideoProcessor = VideoProcessorAdvanced

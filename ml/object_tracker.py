"""
🎬 Advanced Object Tracker with Segmentation
============================================
Использует SAM 2 для сегментации произвольных объектов
и создает красивую визуализацию с траекториями движения.

Author: AI Assistant
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from collections import deque
import colorsys
import time
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 ВИЗУАЛЬНЫЕ НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VisualizationConfig:
    """Конфигурация визуализации"""
    # Цветовая палитра (генерируется автоматически)
    num_colors: int = 20
    
    # Траектории
    trail_length: int = 50  # Длина "хвоста" траектории
    trail_thickness: int = 3
    trail_fade: bool = True  # Затухание траектории
    
    # Маски
    mask_alpha: float = 0.4  # Прозрачность масок
    mask_outline: bool = True
    outline_thickness: int = 2
    
    # Bounding boxes
    show_bbox: bool = True
    bbox_thickness: int = 2
    
    # Центроиды
    show_centroid: bool = True
    centroid_radius: int = 6
    
    # Текст и метки
    show_labels: bool = True
    font_scale: float = 0.7
    font_thickness: int = 2
    
    # Инфо-панель
    show_info_panel: bool = True
    panel_width: int = 300
    panel_alpha: float = 0.85
    
    # Эффекты
    glow_effect: bool = True
    motion_blur_trails: bool = False


def generate_color_palette(n: int) -> List[Tuple[int, int, int]]:
    """Генерирует красивую цветовую палитру"""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.8 + 0.2 * np.sin(i * 0.5)  # Вариация насыщенности
        value = 0.9 + 0.1 * np.cos(i * 0.3)  # Вариация яркости
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


# ═══════════════════════════════════════════════════════════════════════════════
# 📦 СТРУКТУРЫ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedObject:
    """Отслеживаемый объект"""
    id: int
    color: Tuple[int, int, int]
    trajectory: deque = field(default_factory=lambda: deque(maxlen=100))
    masks: List[np.ndarray] = field(default_factory=list)
    bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    centroids: List[Tuple[int, int]] = field(default_factory=list)
    velocities: List[Tuple[float, float]] = field(default_factory=list)
    active: bool = True
    label: str = ""
    confidence: float = 1.0
    
    def add_detection(self, mask: np.ndarray, bbox: Tuple[int, int, int, int], 
                      centroid: Tuple[int, int]):
        """Добавляет новое детекцию"""
        self.masks.append(mask)
        self.bboxes.append(bbox)
        self.centroids.append(centroid)
        self.trajectory.append(centroid)
        
        # Вычисляем скорость
        if len(self.centroids) >= 2:
            prev = self.centroids[-2]
            curr = self.centroids[-1]
            velocity = (curr[0] - prev[0], curr[1] - prev[1])
            self.velocities.append(velocity)
    
    @property
    def current_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        return self.bboxes[-1] if self.bboxes else None
    
    @property
    def current_centroid(self) -> Optional[Tuple[int, int]]:
        return self.centroids[-1] if self.centroids else None
    
    @property
    def current_mask(self) -> Optional[np.ndarray]:
        return self.masks[-1] if self.masks else None
    
    @property
    def average_velocity(self) -> Tuple[float, float]:
        if not self.velocities:
            return (0, 0)
        recent = self.velocities[-10:]  # Последние 10 кадров
        return (
            sum(v[0] for v in recent) / len(recent),
            sum(v[1] for v in recent) / len(recent)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 🖌️ РЕНДЕРИНГ ВИЗУАЛИЗАЦИИ  
# ═══════════════════════════════════════════════════════════════════════════════

class VisualizationRenderer:
    """Красивый рендеринг визуализации"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.colors = generate_color_palette(config.num_colors)
        
    def render_frame(self, frame: np.ndarray, objects: List[TrackedObject],
                     frame_idx: int, total_frames: int, fps: float) -> np.ndarray:
        """Рендерит полный кадр с визуализацией"""
        output = frame.copy()
        
        # 1. Рендерим маски
        output = self._render_masks(output, objects)
        
        # 2. Рендерим траектории
        output = self._render_trajectories(output, objects)
        
        # 3. Рендерим bounding boxes
        if self.config.show_bbox:
            output = self._render_bboxes(output, objects)
        
        # 4. Рендерим центроиды
        if self.config.show_centroid:
            output = self._render_centroids(output, objects)
        
        # 5. Рендерим метки
        if self.config.show_labels:
            output = self._render_labels(output, objects)
        
        # 6. Рендерим инфо-панель
        if self.config.show_info_panel:
            output = self._render_info_panel(output, objects, frame_idx, 
                                             total_frames, fps)
        
        return output
    
    def _render_masks(self, frame: np.ndarray, 
                      objects: List[TrackedObject]) -> np.ndarray:
        """Рендерит сегментационные маски"""
        overlay = frame.copy()
        
        for obj in objects:
            if not obj.active or obj.current_mask is None:
                continue
                
            mask = obj.current_mask
            color = obj.color
            
            # Заливка маски
            overlay[mask > 0] = color
            
            # Контур маски
            if self.config.mask_outline:
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(frame, contours, -1, color, 
                               self.config.outline_thickness)
        
        # Смешиваем с оригиналом
        result = cv2.addWeighted(overlay, self.config.mask_alpha, 
                                 frame, 1 - self.config.mask_alpha, 0)
        return result
    
    def _render_trajectories(self, frame: np.ndarray,
                             objects: List[TrackedObject]) -> np.ndarray:
        """Рендерит траектории движения с эффектом затухания"""
        for obj in objects:
            if not obj.active or len(obj.trajectory) < 2:
                continue
            
            points = list(obj.trajectory)
            color = obj.color
            
            for i in range(1, len(points)):
                if self.config.trail_fade:
                    # Эффект затухания
                    alpha = i / len(points)
                    faded_color = tuple(int(c * alpha) for c in color)
                    thickness = max(1, int(self.config.trail_thickness * alpha))
                else:
                    faded_color = color
                    thickness = self.config.trail_thickness
                
                pt1 = tuple(map(int, points[i - 1]))
                pt2 = tuple(map(int, points[i]))
                
                # Glow эффект
                if self.config.glow_effect:
                    # Внешнее свечение
                    glow_color = tuple(int(c * 0.3) for c in faded_color)
                    cv2.line(frame, pt1, pt2, glow_color, thickness + 4)
                
                cv2.line(frame, pt1, pt2, faded_color, thickness)
        
        return frame
    
    def _render_bboxes(self, frame: np.ndarray,
                       objects: List[TrackedObject]) -> np.ndarray:
        """Рендерит bounding boxes"""
        for obj in objects:
            if not obj.active or obj.current_bbox is None:
                continue
            
            x, y, w, h = obj.current_bbox
            color = obj.color
            
            # Стильный bbox с закругленными углами
            corner_length = min(30, w // 4, h // 4)
            thickness = self.config.bbox_thickness
            
            # Верхний левый угол
            cv2.line(frame, (x, y), (x + corner_length, y), color, thickness)
            cv2.line(frame, (x, y), (x, y + corner_length), color, thickness)
            
            # Верхний правый угол
            cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness)
            cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness)
            
            # Нижний левый угол
            cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness)
            cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness)
            
            # Нижний правый угол
            cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)
        
        return frame
    
    def _render_centroids(self, frame: np.ndarray,
                          objects: List[TrackedObject]) -> np.ndarray:
        """Рендерит центроиды объектов"""
        for obj in objects:
            if not obj.active or obj.current_centroid is None:
                continue
            
            center = tuple(map(int, obj.current_centroid))
            color = obj.color
            radius = self.config.centroid_radius
            
            # Внешнее кольцо
            cv2.circle(frame, center, radius + 3, (255, 255, 255), 2)
            # Внутренний круг
            cv2.circle(frame, center, radius, color, -1)
            # Белая точка в центре
            cv2.circle(frame, center, 2, (255, 255, 255), -1)
        
        return frame
    
    def _render_labels(self, frame: np.ndarray,
                       objects: List[TrackedObject]) -> np.ndarray:
        """Рендерит текстовые метки"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for obj in objects:
            if not obj.active or obj.current_bbox is None:
                continue
            
            x, y, w, h = obj.current_bbox
            color = obj.color
            
            # Текст метки
            label = obj.label if obj.label else f"Object {obj.id}"
            
            # Вычисляем скорость
            vel = obj.average_velocity
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            if speed > 0:
                label += f" | v={speed:.1f}px/f"
            
            # Размер текста
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, self.config.font_scale, self.config.font_thickness
            )
            
            # Фон для текста
            padding = 5
            bg_rect = (x, y - text_h - 2 * padding, 
                      text_w + 2 * padding, text_h + 2 * padding)
            
            # Полупрозрачный фон
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (bg_rect[0], bg_rect[1]),
                         (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]),
                         color, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Текст
            cv2.putText(frame, label, 
                       (x + padding, y - padding),
                       font, self.config.font_scale, (255, 255, 255),
                       self.config.font_thickness)
        
        return frame
    
    def _render_info_panel(self, frame: np.ndarray, objects: List[TrackedObject],
                           frame_idx: int, total_frames: int, 
                           fps: float) -> np.ndarray:
        """Рендерит информационную панель"""
        h, w = frame.shape[:2]
        panel_w = self.config.panel_width
        
        # Создаем панель
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        
        # Градиентный фон
        for i in range(panel_w):
            alpha = 1.0 - (i / panel_w) * 0.3
            panel[:, i] = (int(25 * alpha), int(25 * alpha), int(30 * alpha))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 40
        line_height = 35
        
        # Заголовок
        cv2.putText(panel, "OBJECT TRACKER", (20, y_offset),
                   font, 0.8, (100, 200, 255), 2)
        y_offset += 15
        
        # Линия-разделитель
        cv2.line(panel, (20, y_offset), (panel_w - 20, y_offset), 
                (60, 60, 70), 2)
        y_offset += line_height
        
        # Статистика
        active_objects = sum(1 for o in objects if o.active)
        
        stats = [
            ("Frame", f"{frame_idx + 1}/{total_frames}"),
            ("FPS", f"{fps:.1f}"),
            ("Objects", f"{active_objects}"),
            ("Progress", f"{(frame_idx + 1) / total_frames * 100:.1f}%"),
        ]
        
        for label, value in stats:
            cv2.putText(panel, label, (20, y_offset),
                       font, 0.5, (150, 150, 160), 1)
            cv2.putText(panel, value, (120, y_offset),
                       font, 0.5, (220, 220, 230), 1)
            y_offset += 25
        
        y_offset += 20
        
        # Список объектов
        cv2.putText(panel, "TRACKED OBJECTS", (20, y_offset),
                   font, 0.6, (100, 200, 255), 1)
        y_offset += 10
        cv2.line(panel, (20, y_offset), (panel_w - 20, y_offset), 
                (60, 60, 70), 1)
        y_offset += 25
        
        for obj in objects[:10]:  # Максимум 10 объектов в панели
            if not obj.active:
                continue
            
            # Цветной маркер
            cv2.circle(panel, (30, y_offset - 5), 8, obj.color, -1)
            cv2.circle(panel, (30, y_offset - 5), 8, (255, 255, 255), 1)
            
            # Информация об объекте
            label = obj.label if obj.label else f"Object {obj.id}"
            cv2.putText(panel, label[:15], (50, y_offset),
                       font, 0.45, (200, 200, 210), 1)
            
            # Скорость
            vel = obj.average_velocity
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            cv2.putText(panel, f"v: {speed:.1f}", (50, y_offset + 18),
                       font, 0.35, (120, 120, 130), 1)
            
            y_offset += 50
        
        # Progress bar
        progress_y = h - 50
        progress_w = panel_w - 40
        progress = (frame_idx + 1) / total_frames
        
        cv2.rectangle(panel, (20, progress_y), (20 + progress_w, progress_y + 10),
                     (40, 40, 50), -1)
        cv2.rectangle(panel, (20, progress_y), 
                     (20 + int(progress_w * progress), progress_y + 10),
                     (100, 200, 255), -1)
        
        # Объединяем с основным кадром
        # Создаем расширенный кадр
        result = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
        result[:, :w] = frame
        
        # Добавляем панель с прозрачностью
        panel_area = result[:, w:]
        cv2.addWeighted(panel, self.config.panel_alpha, 
                       panel_area, 1 - self.config.panel_alpha, 0, panel_area)
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 ТРЕКИНГ С ИСПОЛЬЗОВАНИЕМ OPENCV (БАЗОВЫЙ)
# ═══════════════════════════════════════════════════════════════════════════════

class OpenCVTracker:
    """
    Базовый трекер на OpenCV для случаев без SAM.
    Использует комбинацию:
    - Background subtraction для детекции движения
    - Optical flow для трекинга
    """
    
    def __init__(self, min_area: int = 500, max_objects: int = 20):
        self.min_area = min_area
        self.max_objects = max_objects
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.objects: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.colors = generate_color_palette(50)
        
    def process_frame(self, frame: np.ndarray) -> List[TrackedObject]:
        """Обрабатывает кадр и возвращает отслеживаемые объекты"""
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Морфологические операции для очистки
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Удаляем тени (они имеют значение 127)
        fg_mask[fg_mask == 127] = 0
        
        # Находим контуры
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Фильтруем по площади и создаем детекции
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Создаем маску для этого объекта
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Центроид
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            detections.append({
                'bbox': (x, y, w, h),
                'centroid': (cx, cy),
                'mask': mask,
                'area': area
            })
        
        # Сопоставляем детекции с существующими объектами
        self._match_detections(detections)
        
        return list(self.objects.values())
    
    def _match_detections(self, detections: List[Dict]):
        """Сопоставляет детекции с существующими объектами"""
        if not detections:
            # Помечаем все объекты как неактивные
            for obj in self.objects.values():
                obj.active = False
            return
        
        if not self.objects:
            # Создаем новые объекты для всех детекций
            for det in detections[:self.max_objects]:
                self._create_object(det)
            return
        
        # Простое сопоставление по расстоянию
        active_objects = [o for o in self.objects.values() if o.active]
        used_detections = set()
        
        for obj in active_objects:
            if obj.current_centroid is None:
                continue
            
            min_dist = float('inf')
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                
                dist = np.sqrt(
                    (obj.current_centroid[0] - det['centroid'][0])**2 +
                    (obj.current_centroid[1] - det['centroid'][1])**2
                )
                
                if dist < min_dist and dist < 100:  # Порог расстояния
                    min_dist = dist
                    best_det_idx = i
            
            if best_det_idx >= 0:
                det = detections[best_det_idx]
                obj.add_detection(det['mask'], det['bbox'], det['centroid'])
                used_detections.add(best_det_idx)
            else:
                obj.active = False
        
        # Создаем новые объекты для несопоставленных детекций
        for i, det in enumerate(detections):
            if i not in used_detections:
                if len(self.objects) < self.max_objects:
                    self._create_object(det)
    
    def _create_object(self, detection: Dict) -> TrackedObject:
        """Создает новый отслеживаемый объект"""
        obj = TrackedObject(
            id=self.next_id,
            color=self.colors[self.next_id % len(self.colors)]
        )
        obj.add_detection(
            detection['mask'],
            detection['bbox'],
            detection['centroid']
        )
        self.objects[self.next_id] = obj
        self.next_id += 1
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 ТРЕКИНГ С SAM 2 (ПРОДВИНУТЫЙ)
# ═══════════════════════════════════════════════════════════════════════════════

class SAM2Tracker:
    """
    Продвинутый трекер с использованием SAM 2.
    Требует установки: pip install git+https://github.com/facebookresearch/segment-anything-2.git
    """
    
    def __init__(self, model_size: str = "large"):
        """
        Args:
            model_size: 'tiny', 'small', 'base_plus', 'large'
        """
        self.model_size = model_size
        self.predictor = None
        self.objects: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.colors = generate_color_palette(50)
        self.initialized = False
        
    def _load_model(self):
        """Загружает модель SAM 2"""
        try:
            from sam2.build_sam import build_sam2_video_predictor
            
            # Конфигурации моделей
            model_configs = {
                'tiny': 'sam2_hiera_t.yaml',
                'small': 'sam2_hiera_s.yaml',
                'base_plus': 'sam2_hiera_b+.yaml',
                'large': 'sam2_hiera_l.yaml'
            }
            
            checkpoint_urls = {
                'tiny': 'facebook/sam2-hiera-tiny',
                'small': 'facebook/sam2-hiera-small',
                'base_plus': 'facebook/sam2-hiera-base-plus',
                'large': 'facebook/sam2-hiera-large'
            }
            
            config = model_configs.get(self.model_size, model_configs['large'])
            checkpoint = checkpoint_urls.get(self.model_size, checkpoint_urls['large'])
            
            self.predictor = build_sam2_video_predictor(config, checkpoint)
            self.initialized = True
            print(f"✓ SAM 2 ({self.model_size}) загружена успешно")
            
        except ImportError:
            print("⚠ SAM 2 не установлена. Используйте:")
            print("  pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            raise
    
    def init_video(self, video_path: str):
        """Инициализирует видео для трекинга"""
        if not self.initialized:
            self._load_model()
        
        # SAM 2 video predictor initialization
        self.inference_state = self.predictor.init_state(video_path=video_path)
        
    def add_point_prompt(self, frame_idx: int, points: List[Tuple[int, int]], 
                         labels: List[int], object_id: Optional[int] = None):
        """
        Добавляет точечный промпт для сегментации
        
        Args:
            frame_idx: Индекс кадра
            points: Список точек [(x, y), ...]
            labels: Список меток [1 для объекта, 0 для фона]
            object_id: ID объекта (если None - создается новый)
        """
        if object_id is None:
            object_id = self.next_id
            self.next_id += 1
        
        points_np = np.array(points, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int32)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            points=points_np,
            labels=labels_np,
        )
        
        return object_id
    
    def add_box_prompt(self, frame_idx: int, box: Tuple[int, int, int, int],
                       object_id: Optional[int] = None):
        """
        Добавляет box промпт для сегментации
        
        Args:
            frame_idx: Индекс кадра
            box: Bounding box (x1, y1, x2, y2)
            object_id: ID объекта (если None - создается новый)
        """
        if object_id is None:
            object_id = self.next_id
            self.next_id += 1
        
        box_np = np.array(box, dtype=np.float32)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            box=box_np,
        )
        
        return object_id
    
    def propagate(self):
        """Распространяет сегментацию на все кадры видео"""
        video_segments = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in \
                self.predictor.propagate_in_video(self.inference_state):
            
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        return video_segments


# ═══════════════════════════════════════════════════════════════════════════════
# 🦴 YOLO POSE ESTIMATOR (ДЛЯ СКЕЛЕТОВ)
# ═══════════════════════════════════════════════════════════════════════════════

class YOLOPoseEstimator:
    """Оценка позы с помощью YOLO-Pose"""
    
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Голова
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Руки
        (5, 11), (6, 12), (11, 12),  # Торс
        (11, 13), (13, 15), (12, 14), (14, 16)  # Ноги
    ]
    
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, model_size: str = 'n'):
        """
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        """
        self.model_size = model_size
        self.model = None
        self.colors = generate_color_palette(20)
        
    def _load_model(self):
        """Загружает модель YOLO-Pose"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(f'yolov8{self.model_size}-pose.pt')
            print(f"✓ YOLOv8-Pose ({self.model_size}) загружена успешно")
        except ImportError:
            print("⚠ ultralytics не установлена. Используйте:")
            print("  pip install ultralytics")
            raise
    
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Обрабатывает кадр и возвращает позы"""
        if self.model is None:
            self._load_model()
        
        results = self.model(frame, verbose=False)
        poses = []
        
        for result in results:
            if result.keypoints is None:
                continue
            
            keypoints = result.keypoints.xy.cpu().numpy()
            confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
            
            for i, kpts in enumerate(keypoints):
                pose = {
                    'keypoints': kpts,
                    'confidences': confidences[i] if confidences is not None else None,
                    'bbox': boxes[i] if boxes is not None else None
                }
                poses.append(pose)
        
        return poses
    
    def draw_skeleton(self, frame: np.ndarray, poses: List[Dict], 
                      color_idx: int = 0) -> np.ndarray:
        """Рисует скелеты на кадре"""
        for pose_idx, pose in enumerate(poses):
            color = self.colors[(color_idx + pose_idx) % len(self.colors)]
            kpts = pose['keypoints']
            confs = pose.get('confidences')
            
            # Рисуем соединения
            for i, j in self.SKELETON_CONNECTIONS:
                if i < len(kpts) and j < len(kpts):
                    pt1 = tuple(map(int, kpts[i]))
                    pt2 = tuple(map(int, kpts[j]))
                    
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        # Проверяем confidence
                        if confs is not None:
                            if confs[i] < 0.5 or confs[j] < 0.5:
                                continue
                        
                        cv2.line(frame, pt1, pt2, color, 2)
            
            # Рисуем точки
            for i, pt in enumerate(kpts):
                pt = tuple(map(int, pt))
                if pt[0] > 0 and pt[1] > 0:
                    if confs is not None and confs[i] < 0.5:
                        continue
                    
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1)
                    cv2.circle(frame, pt, 3, color, -1)
        
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 ГЛАВНЫЙ ПРОЦЕССОР ВИДЕО
# ═══════════════════════════════════════════════════════════════════════════════

class VideoProcessor:
    """Главный класс для обработки видео"""
    
    def __init__(self, tracker_type: str = 'opencv', 
                 vis_config: Optional[VisualizationConfig] = None):
        """
        Args:
            tracker_type: 'opencv', 'sam2'
            vis_config: Конфигурация визуализации
        """
        self.tracker_type = tracker_type
        self.vis_config = vis_config or VisualizationConfig()
        self.renderer = VisualizationRenderer(self.vis_config)
        
        # Инициализация трекера
        if tracker_type == 'opencv':
            self.tracker = OpenCVTracker()
        elif tracker_type == 'sam2':
            self.tracker = SAM2Tracker()
        else:
            raise ValueError(f"Неизвестный тип трекера: {tracker_type}")
        
        # Опциональный pose estimator
        self.pose_estimator = None
        
    def enable_pose_estimation(self, model_size: str = 'n'):
        """Включает оценку позы"""
        self.pose_estimator = YOLOPoseEstimator(model_size)
        
    def process_video(self, input_path: str, output_path: str,
                      show_preview: bool = True, 
                      max_frames: Optional[int] = None):
        """
        Обрабатывает видео
        
        Args:
            input_path: Путь к входному видео
            output_path: Путь к выходному видео
            show_preview: Показывать превью во время обработки
            max_frames: Максимальное количество кадров (None = все)
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {input_path}")
        
        # Параметры видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        # Подготовка выходного видео
        output_width = width + (self.vis_config.panel_width if self.vis_config.show_info_panel else 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))
        
        print(f"\n{'='*60}")
        print(f"🎬 ОБРАБОТКА ВИДЕО")
        print(f"{'='*60}")
        print(f"📁 Вход:  {input_path}")
        print(f"📁 Выход: {output_path}")
        print(f"📐 Разрешение: {width}x{height}")
        print(f"🎞  FPS: {fps}")
        print(f"📊 Кадров: {total_frames}")
        print(f"🔧 Трекер: {self.tracker_type}")
        print(f"{'='*60}\n")
        
        frame_times = []
        
        with tqdm(total=total_frames, desc="Обработка", unit="кадр") as pbar:
            frame_idx = 0
            
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Трекинг объектов
                objects = self.tracker.process_frame(frame)
                
                # Рисуем позы если включено
                if self.pose_estimator:
                    poses = self.pose_estimator.process_frame(frame)
                    frame = self.pose_estimator.draw_skeleton(frame, poses)
                
                # Вычисляем FPS
                if frame_times:
                    current_fps = 1.0 / np.mean(frame_times[-30:])
                else:
                    current_fps = fps
                
                # Рендерим визуализацию
                output_frame = self.renderer.render_frame(
                    frame, objects, frame_idx, total_frames, current_fps
                )
                
                # Записываем кадр
                out.write(output_frame)
                
                # Показываем превью
                if show_preview:
                    preview = cv2.resize(output_frame, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow('Object Tracker Preview', preview)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⚠ Обработка прервана пользователем")
                        break
                    elif key == ord('p'):
                        cv2.waitKey(0)  # Пауза
                
                # Замеряем время
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                
                pbar.update(1)
                frame_idx += 1
        
        # Очистка
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Статистика
        avg_time = np.mean(frame_times) if frame_times else 0
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"✅ ОБРАБОТКА ЗАВЕРШЕНА")
        print(f"{'='*60}")
        print(f"⏱  Среднее время на кадр: {avg_time*1000:.1f} мс")
        print(f"🚀 Средний FPS обработки: {avg_fps:.1f}")
        print(f"📁 Результат сохранен: {output_path}")
        print(f"{'='*60}\n")
        
        return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 ГЛАВНАЯ ФУНКЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='🎬 Object Tracker with Beautiful Visualization')
    parser.add_argument('input', type=str, help='Путь к входному видео')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Путь к выходному видео (по умолчанию: input_tracked.mp4)')
    parser.add_argument('-t', '--tracker', type=str, default='opencv',
                       choices=['opencv', 'sam2'],
                       help='Тип трекера (по умолчанию: opencv)')
    parser.add_argument('--pose', action='store_true',
                       help='Включить оценку позы (YOLO)')
    parser.add_argument('--pose-model', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Размер модели YOLO-Pose')
    parser.add_argument('--no-preview', action='store_true',
                       help='Отключить превью')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Максимальное количество кадров')
    parser.add_argument('--no-panel', action='store_true',
                       help='Отключить информационную панель')
    parser.add_argument('--trail-length', type=int, default=50,
                       help='Длина траектории')
    
    args = parser.parse_args()
    
    # Путь к выходному файлу
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_tracked.mp4")
    
    # Конфигурация визуализации
    vis_config = VisualizationConfig(
        show_info_panel=not args.no_panel,
        trail_length=args.trail_length
    )
    
    # Создаем процессор
    processor = VideoProcessor(
        tracker_type=args.tracker,
        vis_config=vis_config
    )
    
    # Включаем pose estimation если нужно
    if args.pose:
        processor.enable_pose_estimation(args.pose_model)
    
    # Обрабатываем видео
    processor.process_video(
        input_path=args.input,
        output_path=args.output,
        show_preview=not args.no_preview,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()


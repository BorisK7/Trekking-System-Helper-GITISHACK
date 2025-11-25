#!/usr/bin/env python3
"""
🎭 Advanced Theater Scene Tracker - ПЛАН МАКСИМУМ
===================================================
Модульная система трекинга с полным набором фильтров:

1. Пороговые фильтры (яркость, насыщенность, контраст)
2. Фильтры по размеру и форме (area, aspect ratio, solidity)
3. Temporal фильтры (confirmation, lost frames, velocity clamping)
4. Kalman filter (сглаживание траекторий)
5. YOLO verification (семантическая проверка)
6. Optical flow consistency (консистентность движения)
7. Motion history (паттерны движения)

Запуск:
    python advanced_tracker.py test.mp4
    python advanced_tracker.py test.mp4 --yolo          # С YOLO верификацией
    python advanced_tracker.py test.mp4 --debug         # Debug режим
    python advanced_tracker.py test.mp4 --config strict # Строгие фильтры
"""

import cv2
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Callable
from abc import ABC, abstractmethod
import colorsys
import time


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FilterConfig:
    """Конфигурация всех фильтров"""
    
    # === Пороговые фильтры ===
    brightness_min: int = 15           # Минимальная яркость (отсекаем глубокие тени)
    brightness_max: int = 245          # Максимальная яркость (отсекаем прожекторы)
    saturation_min: int = 10           # Минимальная насыщенность
    contrast_threshold: float = 0.3    # Порог локального контраста
    
    # === Фильтры по размеру ===
    min_area: int = 1500               # Минимальная площадь объекта (px²)
    max_area: int = 200000             # Максимальная площадь объекта (px²)
    max_area_ratio: float = 0.25       # Максимум % от кадра
    
    # === Фильтры по форме ===
    min_aspect_ratio: float = 0.2      # Минимальное соотношение сторон
    max_aspect_ratio: float = 5.0      # Максимальное соотношение сторон
    min_solidity: float = 0.3          # Минимальная "солидность" (заполненность)
    min_extent: float = 0.2            # Минимальный extent (area / bbox_area)
    
    # === Temporal фильтры ===
    confirm_frames: int = 3            # Кадров для подтверждения нового объекта
    lost_frames_max: int = 15          # Кадров до удаления потерянного объекта
    max_velocity: float = 100.0        # Максимальная скорость (px/frame)
    velocity_smoothing: float = 0.7    # Сглаживание скорости (0-1)
    position_smoothing: float = 0.8    # Сглаживание позиции (0-1)
    
    # === Matching ===
    max_match_distance: float = 120.0  # Максимальное расстояние для сопоставления
    
    # === Kalman filter ===
    use_kalman: bool = True            # Использовать Kalman filter
    kalman_process_noise: float = 0.03 # Шум процесса
    kalman_measurement_noise: float = 0.1  # Шум измерения
    
    # === YOLO verification ===
    use_yolo: bool = False             # Использовать YOLO для верификации
    yolo_confidence: float = 0.3       # Минимальная confidence YOLO
    yolo_every_n_frames: int = 3       # Запускать YOLO каждые N кадров
    
    # === Optical flow ===
    use_optical_flow: bool = True      # Использовать optical flow
    flow_consistency_threshold: float = 0.5  # Порог консистентности
    
    @classmethod
    def relaxed(cls) -> 'FilterConfig':
        """Мягкие настройки (больше детекций, больше шума)"""
        return cls(
            min_area=800,
            confirm_frames=2,
            lost_frames_max=25,
            max_velocity=150,
            max_match_distance=150
        )
    
    @classmethod
    def strict(cls) -> 'FilterConfig':
        """Строгие настройки (меньше шума, может пропустить объекты)"""
        return cls(
            min_area=2500,
            confirm_frames=5,
            lost_frames_max=10,
            max_velocity=80,
            max_match_distance=80,
            min_solidity=0.4,
            brightness_max=235
        )
    
    @classmethod
    def theater(cls) -> 'FilterConfig':
        """Оптимизировано для театра"""
        return cls(
            brightness_max=240,
            saturation_min=5,
            min_area=2000,
            max_area_ratio=0.20,
            min_aspect_ratio=0.25,
            max_aspect_ratio=4.0,
            confirm_frames=4,
            lost_frames_max=20,
            max_velocity=90,
            use_optical_flow=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 БАЗОВЫЕ ФИЛЬТРЫ
# ═══════════════════════════════════════════════════════════════════════════════

class BaseFilter(ABC):
    """Базовый класс для всех фильтров"""
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.enabled = True
    
    @abstractmethod
    def apply(self, data: Any) -> Any:
        """Применяет фильтр к данным"""
        pass
    
    @property
    def name(self) -> str:
        return self.__class__.__name__


class BrightnessFilter(BaseFilter):
    """Фильтр по яркости - отсекает прожекторы и глубокие тени"""
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Возвращает маску валидных пикселей (не слишком ярких/тёмных)
        """
        if not self.enabled:
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        # Конвертируем в grayscale или берём V из HSV
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Создаём маску
        mask = np.ones_like(gray) * 255
        mask[gray < self.config.brightness_min] = 0   # Слишком тёмные
        mask[gray > self.config.brightness_max] = 0   # Слишком яркие
        
        return mask


class SaturationFilter(BaseFilter):
    """Фильтр по насыщенности - прожекторы обычно белые (низкая насыщенность)"""
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Возвращает маску пикселей с достаточной насыщенностью
        """
        if not self.enabled:
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # Пиксели с низкой насыщенностью могут быть прожекторами
        # НО также могут быть белой одеждой, поэтому комбинируем с яркостью
        value = hsv[:, :, 2]
        
        mask = np.ones_like(saturation) * 255
        # Отсекаем только если И низкая насыщенность И высокая яркость (прожектор)
        spotlight_mask = (saturation < self.config.saturation_min) & (value > 230)
        mask[spotlight_mask] = 0
        
        return mask


class SizeFilter(BaseFilter):
    """Фильтр по размеру контура"""
    
    def apply(self, contours: List[np.ndarray], frame_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Фильтрует контуры по площади
        """
        if not self.enabled:
            return contours
        
        frame_area = frame_shape[0] * frame_shape[1]
        max_area = min(self.config.max_area, frame_area * self.config.max_area_ratio)
        
        filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.min_area <= area <= max_area:
                filtered.append(contour)
        
        return filtered


class ShapeFilter(BaseFilter):
    """Фильтр по форме контура (aspect ratio, solidity, extent)"""
    
    def apply(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Фильтрует контуры по геометрическим характеристикам
        """
        if not self.enabled:
            return contours
        
        filtered = []
        for contour in contours:
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue
            
            aspect_ratio = w / h
            
            # Aspect ratio check
            if not (self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio):
                continue
            
            # Area
            area = cv2.contourArea(contour)
            if area == 0:
                continue
            
            # Extent (area / bbox_area)
            bbox_area = w * h
            extent = area / bbox_area
            if extent < self.config.min_extent:
                continue
            
            # Solidity (area / convex_hull_area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < self.config.min_solidity:
                    continue
            
            filtered.append(contour)
        
        return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# 📈 KALMAN FILTER
# ═══════════════════════════════════════════════════════════════════════════════

class ObjectKalmanFilter:
    """Kalman filter для сглаживания траектории объекта"""
    
    def __init__(self, initial_pos: Tuple[float, float], 
                 process_noise: float = 0.03,
                 measurement_noise: float = 0.1):
        
        # 4 состояния: x, y, vx, vy
        # 2 измерения: x, y
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Transition matrix (модель движения)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Initial state
        self.kf.statePost = np.array([
            [initial_pos[0]],
            [initial_pos[1]],
            [0],
            [0]
        ], dtype=np.float32)
        
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
    
    def predict(self) -> Tuple[float, float]:
        """Предсказывает следующую позицию"""
        prediction = self.kf.predict()
        return (float(prediction[0]), float(prediction[1]))
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """Обновляет фильтр измерением и возвращает скорректированную позицию"""
        measured = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        corrected = self.kf.correct(measured)
        return (float(corrected[0]), float(corrected[1]))
    
    def get_velocity(self) -> Tuple[float, float]:
        """Возвращает текущую оценку скорости"""
        return (float(self.kf.statePost[2]), float(self.kf.statePost[3]))


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 ОТСЛЕЖИВАЕМЫЙ ОБЪЕКТ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedObject:
    """Отслеживаемый объект с полной информацией"""
    
    id: int
    color: Tuple[int, int, int]
    
    # Позиция и геометрия
    centroid: Optional[Tuple[float, float]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    mask: Optional[np.ndarray] = None
    
    # История
    trajectory: deque = field(default_factory=lambda: deque(maxlen=100))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=20))
    area_history: deque = field(default_factory=lambda: deque(maxlen=20))
    
    # Temporal состояние
    frames_seen: int = 0          # Сколько кадров видели объект
    frames_lost: int = 0          # Сколько кадров не видим
    confirmed: bool = False        # Подтверждён ли объект
    
    # Kalman filter
    kalman: Optional[ObjectKalmanFilter] = None
    
    # YOLO verification
    yolo_confirmed: bool = False   # Подтверждён YOLO
    yolo_class: str = ""           # Класс от YOLO
    yolo_confidence: float = 0.0   # Confidence от YOLO
    
    # Smoothed values
    smoothed_centroid: Optional[Tuple[float, float]] = None
    smoothed_velocity: Tuple[float, float] = (0, 0)
    
    def init_kalman(self, process_noise: float = 0.03, measurement_noise: float = 0.1):
        """Инициализирует Kalman filter"""
        if self.centroid:
            self.kalman = ObjectKalmanFilter(
                self.centroid, process_noise, measurement_noise
            )
    
    def update(self, centroid: Tuple[float, float], 
               bbox: Tuple[int, int, int, int],
               mask: Optional[np.ndarray] = None,
               config: Optional[FilterConfig] = None):
        """Обновляет объект новым измерением"""
        
        # Вычисляем скорость
        if self.centroid is not None:
            raw_velocity = (
                centroid[0] - self.centroid[0],
                centroid[1] - self.centroid[1]
            )
            
            # Velocity clamping
            if config and config.max_velocity > 0:
                speed = np.sqrt(raw_velocity[0]**2 + raw_velocity[1]**2)
                if speed > config.max_velocity:
                    # Ограничиваем скорость
                    scale = config.max_velocity / speed
                    raw_velocity = (raw_velocity[0] * scale, raw_velocity[1] * scale)
            
            self.velocity_history.append(raw_velocity)
            
            # Smoothed velocity
            if config:
                alpha = config.velocity_smoothing
                self.smoothed_velocity = (
                    alpha * self.smoothed_velocity[0] + (1 - alpha) * raw_velocity[0],
                    alpha * self.smoothed_velocity[1] + (1 - alpha) * raw_velocity[1]
                )
        
        # Position smoothing
        if config and self.centroid is not None:
            alpha = config.position_smoothing
            smoothed = (
                alpha * self.centroid[0] + (1 - alpha) * centroid[0],
                alpha * self.centroid[1] + (1 - alpha) * centroid[1]
            )
        else:
            smoothed = centroid
        
        # Kalman update
        if self.kalman:
            self.kalman.predict()
            kalman_pos = self.kalman.update(centroid)
            self.smoothed_centroid = kalman_pos
        else:
            self.smoothed_centroid = smoothed
        
        # Обновляем состояние
        self.centroid = centroid
        self.bbox = bbox
        self.mask = mask
        self.trajectory.append(self.smoothed_centroid or centroid)
        self.area_history.append(bbox[2] * bbox[3] if bbox else 0)
        
        self.frames_seen += 1
        self.frames_lost = 0
    
    def predict_position(self) -> Optional[Tuple[float, float]]:
        """Предсказывает следующую позицию"""
        if self.kalman:
            return self.kalman.predict()
        elif self.centroid and len(self.velocity_history) > 0:
            # Простое линейное предсказание
            return (
                self.centroid[0] + self.smoothed_velocity[0],
                self.centroid[1] + self.smoothed_velocity[1]
            )
        return self.centroid
    
    def get_average_velocity(self) -> Tuple[float, float]:
        """Средняя скорость за последние кадры"""
        if not self.velocity_history:
            return (0, 0)
        vels = list(self.velocity_history)
        return (
            sum(v[0] for v in vels) / len(vels),
            sum(v[1] for v in vels) / len(vels)
        )
    
    def get_speed(self) -> float:
        """Текущая скорость"""
        return np.sqrt(self.smoothed_velocity[0]**2 + self.smoothed_velocity[1]**2)


# ═══════════════════════════════════════════════════════════════════════════════
# 🌊 OPTICAL FLOW
# ═══════════════════════════════════════════════════════════════════════════════

class OpticalFlowAnalyzer:
    """Анализ optical flow для проверки консистентности движения"""
    
    def __init__(self):
        self.prev_gray = None
        self.flow = None
        
        # Параметры для Farneback optical flow
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def compute(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Вычисляет optical flow между текущим и предыдущим кадром"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        
        self.flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, **self.flow_params
        )
        
        self.prev_gray = gray
        return self.flow
    
    def get_flow_at_point(self, point: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """Возвращает вектор flow в точке"""
        if self.flow is None:
            return None
        
        x, y = int(point[0]), int(point[1])
        h, w = self.flow.shape[:2]
        
        if 0 <= x < w and 0 <= y < h:
            return (float(self.flow[y, x, 0]), float(self.flow[y, x, 1]))
        return None
    
    def check_consistency(self, obj: TrackedObject, 
                          threshold: float = 0.5) -> float:
        """
        Проверяет консистентность движения объекта с optical flow.
        Возвращает score от 0 (несовпадение) до 1 (полное совпадение).
        """
        if self.flow is None or obj.centroid is None:
            return 1.0  # Нет данных - считаем валидным
        
        flow_vec = self.get_flow_at_point(obj.centroid)
        if flow_vec is None:
            return 1.0
        
        obj_vel = obj.smoothed_velocity
        
        # Нормализуем векторы
        flow_mag = np.sqrt(flow_vec[0]**2 + flow_vec[1]**2)
        obj_mag = np.sqrt(obj_vel[0]**2 + obj_vel[1]**2)
        
        if flow_mag < 1 and obj_mag < 1:
            return 1.0  # Оба почти неподвижны
        
        if flow_mag < 1 or obj_mag < 1:
            # Один движется, другой нет
            return 0.5
        
        # Косинусное сходство
        dot = flow_vec[0] * obj_vel[0] + flow_vec[1] * obj_vel[1]
        cos_sim = dot / (flow_mag * obj_mag)
        
        # Сходство по магнитуде
        mag_ratio = min(flow_mag, obj_mag) / max(flow_mag, obj_mag)
        
        # Комбинированный score
        score = 0.7 * (cos_sim + 1) / 2 + 0.3 * mag_ratio
        
        return score
    
    def visualize(self, frame: np.ndarray, step: int = 16) -> np.ndarray:
        """Визуализирует optical flow"""
        if self.flow is None:
            return frame
        
        h, w = frame.shape[:2]
        output = frame.copy()
        
        # Рисуем стрелки
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = self.flow[y, x]
                
                # Длина вектора
                mag = np.sqrt(fx*fx + fy*fy)
                if mag < 1:
                    continue
                
                # Цвет по направлению
                angle = np.arctan2(fy, fx)
                hue = int((angle + np.pi) / (2 * np.pi) * 180)
                color = cv2.cvtColor(
                    np.uint8([[[hue, 255, 255]]]), 
                    cv2.COLOR_HSV2BGR
                )[0, 0].tolist()
                
                end_x = int(x + fx * 2)
                end_y = int(y + fy * 2)
                
                cv2.arrowedLine(output, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
        
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 YOLO VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class YOLOVerifier:
    """Верификация объектов с помощью YOLO"""
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence: float = 0.3):
        self.model = None
        self.model_name = model_name
        self.confidence = confidence
        self.last_detections = []
        self.frame_count = 0
        
        # Классы, которые нас интересуют (люди, и возможно некоторые объекты)
        self.target_classes = {0: 'person'}  # COCO class 0 = person
    
    def _load_model(self):
        """Ленивая загрузка модели"""
        if self.model is not None:
            return True
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            print(f"✓ YOLO ({self.model_name}) loaded")
            return True
        except ImportError:
            print("⚠ ultralytics not installed, YOLO verification disabled")
            return False
        except Exception as e:
            print(f"⚠ Failed to load YOLO: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Запускает детекцию на кадре"""
        if not self._load_model():
            return []
        
        results = self.model(frame, verbose=False, conf=self.confidence)
        
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
                cls_id = int(cls)
                if cls_id in self.target_classes:
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    detections.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'centroid': (cx, cy),
                        'class': self.target_classes[cls_id],
                        'confidence': float(conf)
                    })
        
        self.last_detections = detections
        return detections
    
    def verify_object(self, obj: TrackedObject, max_distance: float = 50) -> bool:
        """
        Проверяет, есть ли YOLO детекция рядом с объектом.
        """
        if not self.last_detections or obj.centroid is None:
            return False
        
        for det in self.last_detections:
            dist = np.sqrt(
                (obj.centroid[0] - det['centroid'][0])**2 +
                (obj.centroid[1] - det['centroid'][1])**2
            )
            if dist < max_distance:
                obj.yolo_confirmed = True
                obj.yolo_class = det['class']
                obj.yolo_confidence = det['confidence']
                return True
        
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 ГЛАВНЫЙ ТРЕКЕР
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedTracker:
    """
    Продвинутый трекер с полным набором фильтров.
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig.theater()
        
        # Фильтры
        self.brightness_filter = BrightnessFilter(self.config)
        self.saturation_filter = SaturationFilter(self.config)
        self.size_filter = SizeFilter(self.config)
        self.shape_filter = ShapeFilter(self.config)
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=400,
            varThreshold=40,
            detectShadows=True
        )
        
        # Optical flow
        self.flow_analyzer = OpticalFlowAnalyzer() if self.config.use_optical_flow else None
        
        # YOLO verifier
        self.yolo_verifier = YOLOVerifier() if self.config.use_yolo else None
        
        # Объекты
        self.objects: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.colors = self._generate_colors(50)
        
        # Статистика
        self.stats = {
            'total_detections': 0,
            'filtered_brightness': 0,
            'filtered_size': 0,
            'filtered_shape': 0,
            'filtered_temporal': 0,
            'confirmed_objects': 0
        }
        
        self.frame_count = 0
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Генерирует яркую палитру"""
        colors = []
        for i in range(n):
            hue = (i * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            colors.append(tuple(int(c * 255) for c in rgb[::-1]))
        return colors
    
    def process_frame(self, frame: np.ndarray) -> List[TrackedObject]:
        """
        Обрабатывает кадр через весь pipeline фильтров.
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 1: Preprocessing & Background Subtraction
        # ═══════════════════════════════════════════════════════════════════════
        
        # Применяем пороговые фильтры
        brightness_mask = self.brightness_filter.apply(frame)
        saturation_mask = self.saturation_filter.apply(frame)
        
        # Комбинируем маски
        combined_mask = cv2.bitwise_and(brightness_mask, saturation_mask)
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask[fg_mask == 127] = 0  # Удаляем тени
        
        # Применяем пороговые маски
        fg_mask = cv2.bitwise_and(fg_mask, combined_mask)
        
        # Морфология
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 2: Contour Detection & Spatial Filtering
        # ═══════════════════════════════════════════════════════════════════════
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        self.stats['total_detections'] = len(contours)
        
        # Фильтр по размеру
        contours = self.size_filter.apply(contours, (h, w))
        self.stats['filtered_size'] = self.stats['total_detections'] - len(contours)
        
        # Фильтр по форме
        before_shape = len(contours)
        contours = self.shape_filter.apply(contours)
        self.stats['filtered_shape'] = before_shape - len(contours)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 3: Create Detections
        # ═══════════════════════════════════════════════════════════════════════
        
        detections = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + cw / 2, y + ch / 2
            
            obj_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(obj_mask, [contour], -1, 255, -1)
            
            detections.append({
                'centroid': (cx, cy),
                'bbox': (x, y, cw, ch),
                'mask': obj_mask,
                'area': area
            })
        
        # Сортируем по площади
        detections.sort(key=lambda d: d['area'], reverse=True)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 4: Optical Flow
        # ═══════════════════════════════════════════════════════════════════════
        
        if self.flow_analyzer:
            self.flow_analyzer.compute(frame)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 5: YOLO Verification (каждые N кадров)
        # ═══════════════════════════════════════════════════════════════════════
        
        if self.yolo_verifier and self.frame_count % self.config.yolo_every_n_frames == 0:
            self.yolo_verifier.detect(frame)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 6: Temporal Matching & Tracking
        # ═══════════════════════════════════════════════════════════════════════
        
        # Увеличиваем lost_frames для всех объектов
        for obj in self.objects.values():
            obj.frames_lost += 1
        
        # Сопоставляем детекции с объектами
        self._match_detections(detections)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 7: Temporal Filtering & Cleanup
        # ═══════════════════════════════════════════════════════════════════════
        
        # Подтверждаем объекты
        for obj in self.objects.values():
            if not obj.confirmed and obj.frames_seen >= self.config.confirm_frames:
                obj.confirmed = True
                self.stats['confirmed_objects'] += 1
        
        # Проверяем optical flow consistency
        if self.flow_analyzer:
            for obj in self.objects.values():
                if obj.confirmed:
                    consistency = self.flow_analyzer.check_consistency(obj)
                    # Можно использовать для фильтрации или просто логировать
        
        # YOLO verification
        if self.yolo_verifier:
            for obj in self.objects.values():
                if obj.confirmed and not obj.yolo_confirmed:
                    self.yolo_verifier.verify_object(obj)
        
        # Удаляем потерянные объекты
        self._cleanup_lost_objects()
        
        # Возвращаем только подтверждённые объекты
        return [obj for obj in self.objects.values() if obj.confirmed]
    
    def _match_detections(self, detections: List[Dict]):
        """Сопоставляет детекции с существующими объектами"""
        
        if not detections:
            return
        
        used_detections = set()
        
        # Сначала сопоставляем с существующими объектами
        for obj_id, obj in list(self.objects.items()):
            if obj.centroid is None:
                continue
            
            # Предсказываем позицию
            predicted = obj.predict_position()
            if predicted is None:
                predicted = obj.centroid
            
            min_dist = float('inf')
            best_idx = -1
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                
                # Расстояние до текущей позиции
                dist_current = np.sqrt(
                    (obj.centroid[0] - det['centroid'][0])**2 +
                    (obj.centroid[1] - det['centroid'][1])**2
                )
                
                # Расстояние до предсказанной позиции
                dist_predicted = np.sqrt(
                    (predicted[0] - det['centroid'][0])**2 +
                    (predicted[1] - det['centroid'][1])**2
                )
                
                dist = min(dist_current, dist_predicted)
                
                # Velocity check
                if self.config.max_velocity > 0:
                    velocity_dist = np.sqrt(
                        (obj.centroid[0] - det['centroid'][0])**2 +
                        (obj.centroid[1] - det['centroid'][1])**2
                    )
                    if velocity_dist > self.config.max_velocity * 1.5:
                        continue  # Слишком быстрое перемещение
                
                if dist < min_dist and dist < self.config.max_match_distance:
                    min_dist = dist
                    best_idx = i
            
            if best_idx >= 0:
                det = detections[best_idx]
                obj.update(det['centroid'], det['bbox'], det['mask'], self.config)
                used_detections.add(best_idx)
        
        # Создаём новые объекты для несопоставленных детекций
        for i, det in enumerate(detections):
            if i not in used_detections:
                self._create_object(det)
    
    def _create_object(self, detection: Dict):
        """Создаёт новый объект"""
        obj = TrackedObject(
            id=self.next_id,
            color=self.colors[self.next_id % len(self.colors)]
        )
        
        obj.update(detection['centroid'], detection['bbox'], detection['mask'], self.config)
        
        if self.config.use_kalman:
            obj.init_kalman(
                self.config.kalman_process_noise,
                self.config.kalman_measurement_noise
            )
        
        self.objects[self.next_id] = obj
        self.next_id += 1
    
    def _cleanup_lost_objects(self):
        """Удаляет потерянные объекты"""
        to_remove = []
        
        for obj_id, obj in self.objects.items():
            if obj.frames_lost > self.config.lost_frames_max:
                to_remove.append(obj_id)
            elif not obj.confirmed and obj.frames_lost > 3:
                # Неподтверждённые объекты удаляем быстрее
                to_remove.append(obj_id)
        
        self.stats['filtered_temporal'] = len(to_remove)
        
        for obj_id in to_remove:
            del self.objects[obj_id]


# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedVisualizer:
    """Продвинутая визуализация с debug-режимом"""
    
    def __init__(self, show_debug: bool = False, show_flow: bool = False):
        self.show_debug = show_debug
        self.show_flow = show_flow
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def render(self, frame: np.ndarray, objects: List[TrackedObject],
               tracker: AdvancedTracker, frame_idx: int, 
               total_frames: int, fps: float) -> np.ndarray:
        """Рендерит финальный кадр"""
        
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Optical flow visualization
        if self.show_flow and tracker.flow_analyzer:
            output = tracker.flow_analyzer.visualize(output)
        
        # Маски объектов
        for obj in objects:
            if obj.mask is not None and obj.confirmed:
                overlay = output.copy()
                overlay[obj.mask > 0] = obj.color
                cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
                
                # Контур
                contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, contours, -1, obj.color, 2)
        
        # Траектории
        for obj in objects:
            if len(obj.trajectory) > 1 and obj.confirmed:
                pts = list(obj.trajectory)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    color = tuple(int(c * alpha) for c in obj.color)
                    thickness = max(1, int(3 * alpha))
                    pt1 = tuple(map(int, pts[i-1]))
                    pt2 = tuple(map(int, pts[i]))
                    cv2.line(output, pt1, pt2, color, thickness)
        
        # Bounding boxes и метки
        for obj in objects:
            if obj.bbox and obj.confirmed:
                x, y, bw, bh = obj.bbox
                
                # Стильный bbox
                corner = min(20, bw // 5, bh // 5)
                corners = [
                    [(x, y), (x + corner, y)],
                    [(x, y), (x, y + corner)],
                    [(x + bw, y), (x + bw - corner, y)],
                    [(x + bw, y), (x + bw, y + corner)],
                    [(x, y + bh), (x + corner, y + bh)],
                    [(x, y + bh), (x, y + bh - corner)],
                    [(x + bw, y + bh), (x + bw - corner, y + bh)],
                    [(x + bw, y + bh), (x + bw, y + bh - corner)],
                ]
                for p1, p2 in corners:
                    cv2.line(output, p1, p2, obj.color, 2)
                
                # Метка
                label = f"#{obj.id}"
                if obj.yolo_confirmed:
                    label += f" [{obj.yolo_class}]"
                
                speed = obj.get_speed()
                if speed > 1:
                    label += f" v:{speed:.0f}"
                
                cv2.putText(output, label, (x, y - 8),
                           self.font, 0.5, (0, 0, 0), 3)
                cv2.putText(output, label, (x, y - 8),
                           self.font, 0.5, obj.color, 1)
        
        # Центроиды
        for obj in objects:
            if obj.smoothed_centroid and obj.confirmed:
                center = tuple(map(int, obj.smoothed_centroid))
                cv2.circle(output, center, 6, (255, 255, 255), 2)
                cv2.circle(output, center, 4, obj.color, -1)
        
        # Информационная панель
        self._draw_info_panel(output, objects, tracker, frame_idx, total_frames, fps)
        
        # Debug панель
        if self.show_debug:
            self._draw_debug_panel(output, tracker)
        
        return output
    
    def _draw_info_panel(self, frame: np.ndarray, objects: List[TrackedObject],
                         tracker: AdvancedTracker, frame_idx: int,
                         total_frames: int, fps: float):
        """Рисует информационную панель"""
        h, w = frame.shape[:2]
        
        # Фон панели
        panel_h = 130
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 35
        cv2.putText(frame, "ADVANCED TRACKER", (20, y),
                   self.font, 0.7, (100, 200, 255), 2)
        
        y += 25
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames}", (20, y),
                   self.font, 0.45, (200, 200, 200), 1)
        
        y += 20
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y),
                   self.font, 0.45, (200, 200, 200), 1)
        
        y += 20
        confirmed = len([o for o in objects if o.confirmed])
        unconfirmed = len(tracker.objects) - confirmed
        cv2.putText(frame, f"Objects: {confirmed} confirmed, {unconfirmed} pending", (20, y),
                   self.font, 0.4, (200, 200, 200), 1)
        
        y += 20
        progress = (frame_idx + 1) / total_frames
        bar_w = 260
        cv2.rectangle(frame, (20, y), (20 + bar_w, y + 8), (50, 50, 60), -1)
        cv2.rectangle(frame, (20, y), (20 + int(bar_w * progress), y + 8),
                     (100, 200, 255), -1)
    
    def _draw_debug_panel(self, frame: np.ndarray, tracker: AdvancedTracker):
        """Рисует debug информацию"""
        h, w = frame.shape[:2]
        
        # Фон debug панели (справа)
        panel_w = 220
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w - 10, 10), (w - 10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        x = w - panel_w
        y = 30
        
        cv2.putText(frame, "DEBUG INFO", (x, y),
                   self.font, 0.5, (255, 200, 100), 1)
        
        stats = [
            f"Raw detections: {tracker.stats['total_detections']}",
            f"Filtered (size): {tracker.stats['filtered_size']}",
            f"Filtered (shape): {tracker.stats['filtered_shape']}",
            f"Filtered (temporal): {tracker.stats['filtered_temporal']}",
            f"Total confirmed: {tracker.stats['confirmed_objects']}",
        ]
        
        y += 25
        for stat in stats:
            cv2.putText(frame, stat, (x, y),
                       self.font, 0.35, (180, 180, 180), 1)
            y += 18
        
        # Config info
        y += 10
        cv2.putText(frame, "Config:", (x, y), self.font, 0.4, (255, 200, 100), 1)
        y += 18
        
        config_info = [
            f"Confirm frames: {tracker.config.confirm_frames}",
            f"Max velocity: {tracker.config.max_velocity}",
            f"YOLO: {'ON' if tracker.config.use_yolo else 'OFF'}",
            f"Kalman: {'ON' if tracker.config.use_kalman else 'OFF'}",
        ]
        
        for info in config_info:
            cv2.putText(frame, info, (x, y),
                       self.font, 0.3, (150, 150, 150), 1)
            y += 15


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 ПРОЦЕССОР
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedProcessor:
    """Главный процессор видео"""
    
    def __init__(self, config: Optional[FilterConfig] = None,
                 show_debug: bool = False, show_flow: bool = False):
        self.tracker = AdvancedTracker(config)
        self.visualizer = AdvancedVisualizer(show_debug, show_flow)
    
    def process_video(self, input_path: str, output_path: str,
                      show_preview: bool = True,
                      max_frames: Optional[int] = None):
        """Обрабатывает видео"""
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {input_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total = min(total, max_frames)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        config = self.tracker.config
        
        print(f"\n{'═'*65}")
        print(f"  🎭 ADVANCED THEATER TRACKER - PLAN MAXIMUM")
        print(f"{'═'*65}")
        print(f"  📁 Input:  {input_path}")
        print(f"  📁 Output: {output_path}")
        print(f"  📐 Size:   {width}x{height} @ {fps:.1f} FPS")
        print(f"  📊 Frames: {total}")
        print(f"{'─'*65}")
        print(f"  ⚙️  CONFIG:")
        print(f"     • Brightness filter: {config.brightness_min}-{config.brightness_max}")
        print(f"     • Size filter: {config.min_area}-{config.max_area} px²")
        print(f"     • Temporal: confirm={config.confirm_frames}, lost={config.lost_frames_max}")
        print(f"     • Max velocity: {config.max_velocity} px/frame")
        print(f"     • Kalman filter: {'ON' if config.use_kalman else 'OFF'}")
        print(f"     • YOLO verify: {'ON' if config.use_yolo else 'OFF'}")
        print(f"     • Optical flow: {'ON' if config.use_optical_flow else 'OFF'}")
        print(f"{'═'*65}")
        print(f"\n  Press 'Q' to quit, 'P' to pause, 'D' for debug\n")
        
        frame_times = []
        frame_idx = 0
        show_debug = self.visualizer.show_debug
        
        while frame_idx < total:
            ret, frame = cap.read()
            if not ret:
                break
            
            start = time.time()
            
            # Обработка
            objects = self.tracker.process_frame(frame)
            
            # Визуализация
            current_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else fps
            output_frame = self.visualizer.render(
                frame, objects, self.tracker, frame_idx, total, current_fps
            )
            
            frame_time = time.time() - start
            frame_times.append(frame_time)
            
            out.write(output_frame)
            
            if show_preview:
                preview = cv2.resize(output_frame, (0, 0), fx=0.6, fy=0.6)
                cv2.imshow('Advanced Tracker', preview)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  Interrupted")
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
                elif key == ord('d'):
                    self.visualizer.show_debug = not self.visualizer.show_debug
            
            if frame_idx % 30 == 0:
                progress = (frame_idx + 1) / total * 100
                avg_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
                confirmed = len([o for o in objects if o.confirmed])
                print(f"\r  [{progress:5.1f}%] FPS: {avg_fps:.1f} | Objects: {confirmed}", end="")
            
            frame_idx += 1
        
        print(f"\n\n{'═'*65}")
        print(f"  ✅ PROCESSING COMPLETE")
        print(f"{'═'*65}")
        if frame_times:
            print(f"  ⏱  Avg frame time: {np.mean(frame_times)*1000:.1f} ms")
            print(f"  🚀 Avg FPS: {1.0/np.mean(frame_times):.1f}")
        print(f"  📊 Total objects tracked: {self.tracker.stats['confirmed_objects']}")
        print(f"  💾 Saved: {output_path}")
        print(f"{'═'*65}\n")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🎭 Advanced Theater Tracker - PLAN MAXIMUM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configurations:
  default  - Balanced settings for most cases
  relaxed  - More detections, more noise
  strict   - Less noise, may miss some objects
  theater  - Optimized for theater scenes (recommended)

Examples:
  python advanced_tracker.py test.mp4
  python advanced_tracker.py test.mp4 --config strict
  python advanced_tracker.py test.mp4 --yolo --debug
  python advanced_tracker.py test.mp4 --confirm-frames 5 --max-velocity 60
        """
    )
    
    parser.add_argument('input', help='Input video')
    parser.add_argument('-o', '--output', default=None, help='Output path')
    parser.add_argument('--config', default='theater',
                       choices=['default', 'relaxed', 'strict', 'theater'],
                       help='Preset configuration')
    
    # Filter overrides
    parser.add_argument('--min-area', type=int, default=None)
    parser.add_argument('--max-area', type=int, default=None)
    parser.add_argument('--confirm-frames', type=int, default=None)
    parser.add_argument('--lost-frames', type=int, default=None)
    parser.add_argument('--max-velocity', type=float, default=None)
    parser.add_argument('--brightness-max', type=int, default=None)
    
    # Features
    parser.add_argument('--yolo', action='store_true', help='Enable YOLO verification')
    parser.add_argument('--no-kalman', action='store_true', help='Disable Kalman filter')
    parser.add_argument('--no-flow', action='store_true', help='Disable optical flow')
    
    # Visualization
    parser.add_argument('--debug', action='store_true', help='Show debug info')
    parser.add_argument('--show-flow', action='store_true', help='Visualize optical flow')
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--max-frames', type=int, default=None)
    
    args = parser.parse_args()
    
    # Load config preset
    config_map = {
        'default': FilterConfig(),
        'relaxed': FilterConfig.relaxed(),
        'strict': FilterConfig.strict(),
        'theater': FilterConfig.theater()
    }
    config = config_map[args.config]
    
    # Apply overrides
    if args.min_area is not None:
        config.min_area = args.min_area
    if args.max_area is not None:
        config.max_area = args.max_area
    if args.confirm_frames is not None:
        config.confirm_frames = args.confirm_frames
    if args.lost_frames is not None:
        config.lost_frames_max = args.lost_frames
    if args.max_velocity is not None:
        config.max_velocity = args.max_velocity
    if args.brightness_max is not None:
        config.brightness_max = args.brightness_max
    
    config.use_yolo = args.yolo
    config.use_kalman = not args.no_kalman
    config.use_optical_flow = not args.no_flow
    
    # Output path
    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_advanced.mp4")
    
    # Process
    processor = AdvancedProcessor(
        config=config,
        show_debug=args.debug,
        show_flow=args.show_flow
    )
    
    processor.process_video(
        args.input,
        args.output,
        show_preview=not args.no_preview,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()


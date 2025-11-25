#!/usr/bin/env python3
"""
🎭 Advanced Theater Tracker v2 - С MERGE ОБЪЕКТОВ
===================================================
Добавлено:
- Merge близких объектов
- Merge по схожему вектору движения
- Merge по IoU (пересечению bbox)
- Иерархическая кластеризация детекций

Запуск:
    python advanced_tracker_v2.py test.mp4
    python advanced_tracker_v2.py test.mp4 --merge-distance 100
    python advanced_tracker_v2.py test.mp4 --merge-velocity 0.8
"""

import cv2
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import pdist, squareform
import colorsys
import time


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MergeConfig:
    """Конфигурация слияния объектов"""
    
    # === Merge по расстоянию ===
    merge_distance: float = 80.0       # Максимальное расстояние для merge (px)
    merge_enabled: bool = True
    
    # === Merge по вектору движения ===
    velocity_merge_enabled: bool = True
    velocity_similarity_threshold: float = 0.7  # Косинусное сходство (0-1)
    velocity_distance_factor: float = 1.5       # Увеличиваем допустимое расстояние для объектов с похожим движением
    
    # === Merge по IoU ===
    iou_merge_enabled: bool = True
    iou_threshold: float = 0.1         # Минимальный IoU для merge
    
    # === Merge по вертикали (человек = голова + торс + ноги) ===
    vertical_merge_enabled: bool = True
    vertical_gap_max: float = 50.0     # Максимальный вертикальный разрыв
    horizontal_overlap_min: float = 0.5  # Минимальное горизонтальное перекрытие
    
    # === Кластеризация ===
    use_clustering: bool = True
    cluster_distance_threshold: float = 100.0
    
    # === Морфология перед детекцией ===
    use_morphological_merge: bool = True
    morph_close_size: int = 15         # Размер ядра для close (соединяет близкие области)


@dataclass 
class TrackerConfig:
    """Конфигурация трекера"""
    # Фильтры
    brightness_min: int = 15
    brightness_max: int = 245
    min_area: int = 1500
    max_area: int = 200000
    min_aspect_ratio: float = 0.15
    max_aspect_ratio: float = 6.0
    
    # Temporal
    confirm_frames: int = 3
    lost_frames_max: int = 20
    max_velocity: float = 100.0
    
    # Kalman
    use_kalman: bool = True
    
    # Merge
    merge: MergeConfig = field(default_factory=MergeConfig)


# ═══════════════════════════════════════════════════════════════════════════════
# 🔗 МОДУЛЬ СЛИЯНИЯ ДЕТЕКЦИЙ
# ═══════════════════════════════════════════════════════════════════════════════

class DetectionMerger:
    """
    Модуль слияния детекций в единые объекты.
    Решает проблему когда один человек детектируется как несколько частей.
    """
    
    def __init__(self, config: MergeConfig):
        self.config = config
    
    def merge_detections(self, detections: List[Dict], 
                         prev_velocities: Optional[Dict[int, Tuple[float, float]]] = None) -> List[Dict]:
        """
        Главная функция слияния детекций.
        
        Args:
            detections: Список детекций [{'centroid': (x,y), 'bbox': (x,y,w,h), 'mask': ...}]
            prev_velocities: Словарь предыдущих скоростей {idx: (vx, vy)}
        
        Returns:
            Список объединённых детекций
        """
        if not detections or len(detections) < 2:
            return detections
        
        # Создаём граф связей
        n = len(detections)
        merge_graph = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(detections[i], detections[j], prev_velocities, i, j):
                    merge_graph[i, j] = True
                    merge_graph[j, i] = True
        
        # Находим связные компоненты
        clusters = self._find_connected_components(merge_graph)
        
        # Объединяем детекции в каждом кластере
        merged = []
        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(detections[cluster[0]])
            else:
                merged_det = self._merge_cluster([detections[i] for i in cluster])
                merged.append(merged_det)
        
        return merged
    
    def _should_merge(self, det1: Dict, det2: Dict,
                      velocities: Optional[Dict], idx1: int, idx2: int) -> bool:
        """Определяет, нужно ли объединять две детекции"""
        
        c1, c2 = det1['centroid'], det2['centroid']
        b1, b2 = det1['bbox'], det2['bbox']
        
        # Расстояние между центроидами
        distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        
        # === 1. Merge по расстоянию ===
        if self.config.merge_enabled:
            merge_dist = self.config.merge_distance
            
            # Увеличиваем допустимое расстояние если есть похожая скорость
            if velocities and self.config.velocity_merge_enabled:
                v1 = velocities.get(idx1)
                v2 = velocities.get(idx2)
                if v1 and v2:
                    similarity = self._velocity_similarity(v1, v2)
                    if similarity > self.config.velocity_similarity_threshold:
                        merge_dist *= self.config.velocity_distance_factor
            
            if distance < merge_dist:
                return True
        
        # === 2. Merge по IoU ===
        if self.config.iou_merge_enabled:
            iou = self._compute_iou(b1, b2)
            if iou > self.config.iou_threshold:
                return True
        
        # === 3. Merge по вертикали (части тела) ===
        if self.config.vertical_merge_enabled:
            if self._is_vertical_stack(b1, b2):
                return True
        
        # === 4. Merge по вектору движения ===
        if velocities and self.config.velocity_merge_enabled:
            v1 = velocities.get(idx1)
            v2 = velocities.get(idx2)
            if v1 and v2:
                similarity = self._velocity_similarity(v1, v2)
                # Если скорости очень похожи и объекты не слишком далеко
                if similarity > 0.9 and distance < self.config.merge_distance * 2:
                    return True
        
        return False
    
    def _velocity_similarity(self, v1: Tuple[float, float], 
                            v2: Tuple[float, float]) -> float:
        """
        Вычисляет схожесть векторов движения (косинусное сходство).
        Возвращает значение от 0 до 1.
        """
        mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        # Если оба почти неподвижны
        if mag1 < 2 and mag2 < 2:
            return 1.0
        
        # Если один движется, другой нет
        if mag1 < 2 or mag2 < 2:
            return 0.5
        
        # Косинусное сходство
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_sim = dot / (mag1 * mag2)
        
        # Преобразуем в диапазон [0, 1]
        similarity = (cos_sim + 1) / 2
        
        # Учитываем схожесть по магнитуде
        mag_ratio = min(mag1, mag2) / max(mag1, mag2)
        
        # Комбинируем
        return 0.7 * similarity + 0.3 * mag_ratio
    
    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Вычисляет Intersection over Union для двух bbox"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Координаты пересечения
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_vertical_stack(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """
        Проверяет, являются ли bbox вертикально расположенными частями одного объекта.
        (например, голова над торсом)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Определяем верхний и нижний bbox
        if y1 < y2:
            upper, lower = bbox1, bbox2
        else:
            upper, lower = bbox2, bbox1
        
        ux, uy, uw, uh = upper
        lx, ly, lw, lh = lower
        
        # Вертикальный разрыв между ними
        vertical_gap = ly - (uy + uh)
        
        if vertical_gap < 0 or vertical_gap > self.config.vertical_gap_max:
            return False
        
        # Горизонтальное перекрытие
        overlap_left = max(ux, lx)
        overlap_right = min(ux + uw, lx + lw)
        
        if overlap_right <= overlap_left:
            return False
        
        overlap_width = overlap_right - overlap_left
        min_width = min(uw, lw)
        
        overlap_ratio = overlap_width / min_width
        
        return overlap_ratio >= self.config.horizontal_overlap_min
    
    def _find_connected_components(self, graph: np.ndarray) -> List[List[int]]:
        """Находит связные компоненты в графе"""
        n = len(graph)
        visited = [False] * n
        components = []
        
        def dfs(node: int, component: List[int]):
            visited[node] = True
            component.append(node)
            for neighbor in range(n):
                if graph[node, neighbor] and not visited[neighbor]:
                    dfs(neighbor, component)
        
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, component)
                components.append(component)
        
        return components
    
    def _merge_cluster(self, detections: List[Dict]) -> Dict:
        """Объединяет кластер детекций в одну"""
        
        # Объединяем bbox'ы
        x_min = min(d['bbox'][0] for d in detections)
        y_min = min(d['bbox'][1] for d in detections)
        x_max = max(d['bbox'][0] + d['bbox'][2] for d in detections)
        y_max = max(d['bbox'][1] + d['bbox'][3] for d in detections)
        
        merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Центроид - взвешенный по площади
        total_area = sum(d.get('area', d['bbox'][2] * d['bbox'][3]) for d in detections)
        cx = sum(d['centroid'][0] * d.get('area', d['bbox'][2] * d['bbox'][3]) 
                for d in detections) / total_area
        cy = sum(d['centroid'][1] * d.get('area', d['bbox'][2] * d['bbox'][3]) 
                for d in detections) / total_area
        
        # Объединяем маски
        merged_mask = None
        for d in detections:
            if d.get('mask') is not None:
                if merged_mask is None:
                    merged_mask = d['mask'].copy()
                else:
                    merged_mask = cv2.bitwise_or(merged_mask, d['mask'])
        
        return {
            'centroid': (cx, cy),
            'bbox': merged_bbox,
            'mask': merged_mask,
            'area': total_area,
            'merged_from': len(detections)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 🔗 КЛАСТЕРИЗАЦИЯ ДЕТЕКЦИЙ
# ═══════════════════════════════════════════════════════════════════════════════

class DetectionClusterer:
    """
    Иерархическая кластеризация детекций.
    Альтернативный подход к merge.
    """
    
    def __init__(self, distance_threshold: float = 100.0):
        self.distance_threshold = distance_threshold
    
    def cluster(self, detections: List[Dict], 
                velocities: Optional[Dict[int, Tuple[float, float]]] = None) -> List[Dict]:
        """
        Кластеризует детекции используя иерархическую кластеризацию.
        """
        if len(detections) < 2:
            return detections
        
        # Собираем признаки: [x, y, vx, vy, w, h]
        features = []
        for i, det in enumerate(detections):
            cx, cy = det['centroid']
            w, h = det['bbox'][2], det['bbox'][3]
            
            # Добавляем скорость если есть
            vx, vy = 0, 0
            if velocities and i in velocities:
                vx, vy = velocities[i]
            
            # Нормализуем признаки
            features.append([
                cx / 100,  # position (scaled down)
                cy / 100,
                vx / 10,   # velocity
                vy / 10,
                w / 50,    # size
                h / 50
            ])
        
        features = np.array(features)
        
        # Иерархическая кластеризация
        try:
            clusters = fclusterdata(
                features, 
                t=self.distance_threshold / 100,  # scaled threshold
                criterion='distance',
                method='average'
            )
        except:
            # Если кластеризация не удалась, возвращаем как есть
            return detections
        
        # Группируем детекции по кластерам
        cluster_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(i)
        
        # Объединяем
        merger = DetectionMerger(MergeConfig())
        merged = []
        
        for indices in cluster_groups.values():
            if len(indices) == 1:
                merged.append(detections[indices[0]])
            else:
                cluster_dets = [detections[i] for i in indices]
                merged.append(merger._merge_cluster(cluster_dets))
        
        return merged


# ═══════════════════════════════════════════════════════════════════════════════
# 📈 KALMAN FILTER
# ═══════════════════════════════════════════════════════════════════════════════

class KalmanTracker:
    """Kalman filter для сглаживания траектории"""
    
    def __init__(self, initial_pos: Tuple[float, float]):
        self.kf = cv2.KalmanFilter(4, 2)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        self.kf.statePost = np.array([
            [initial_pos[0]], [initial_pos[1]], [0], [0]
        ], dtype=np.float32)
        
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
    
    def predict(self) -> Tuple[float, float]:
        pred = self.kf.predict()
        return (float(pred[0]), float(pred[1]))
    
    def update(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        measured = np.array([[pos[0]], [pos[1]]], dtype=np.float32)
        corrected = self.kf.correct(measured)
        return (float(corrected[0]), float(corrected[1]))
    
    def get_velocity(self) -> Tuple[float, float]:
        return (float(self.kf.statePost[2]), float(self.kf.statePost[3]))


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 ОТСЛЕЖИВАЕМЫЙ ОБЪЕКТ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedObject:
    """Отслеживаемый объект"""
    
    id: int
    color: Tuple[int, int, int]
    
    centroid: Optional[Tuple[float, float]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    mask: Optional[np.ndarray] = None
    
    trajectory: deque = field(default_factory=lambda: deque(maxlen=100))
    velocity: Tuple[float, float] = (0, 0)
    
    frames_seen: int = 0
    frames_lost: int = 0
    confirmed: bool = False
    
    kalman: Optional[KalmanTracker] = None
    smoothed_pos: Optional[Tuple[float, float]] = None
    
    merged_count: int = 1  # Из скольких детекций собран
    
    def init_kalman(self):
        if self.centroid:
            self.kalman = KalmanTracker(self.centroid)
    
    def update(self, centroid: Tuple[float, float], 
               bbox: Tuple[int, int, int, int],
               mask: Optional[np.ndarray] = None,
               merged_count: int = 1):
        
        # Скорость
        if self.centroid:
            self.velocity = (
                centroid[0] - self.centroid[0],
                centroid[1] - self.centroid[1]
            )
        
        # Kalman
        if self.kalman:
            self.kalman.predict()
            self.smoothed_pos = self.kalman.update(centroid)
        else:
            self.smoothed_pos = centroid
        
        self.centroid = centroid
        self.bbox = bbox
        self.mask = mask
        self.merged_count = merged_count
        
        self.trajectory.append(self.smoothed_pos or centroid)
        self.frames_seen += 1
        self.frames_lost = 0
    
    def predict_position(self) -> Optional[Tuple[float, float]]:
        if self.kalman:
            return self.kalman.predict()
        elif self.centroid:
            return (
                self.centroid[0] + self.velocity[0],
                self.centroid[1] + self.velocity[1]
            )
        return None
    
    def get_speed(self) -> float:
        return np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 ГЛАВНЫЙ ТРЕКЕР V2
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedTrackerV2:
    """
    Продвинутый трекер v2 с объединением фрагментированных детекций.
    """
    
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        
        # Модули
        self.merger = DetectionMerger(self.config.merge)
        self.clusterer = DetectionClusterer(
            self.config.merge.cluster_distance_threshold
        ) if self.config.merge.use_clustering else None
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=40, detectShadows=True
        )
        
        # Объекты
        self.objects: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.colors = self._generate_colors(50)
        
        # История скоростей детекций (для merge)
        self.prev_detection_velocities: Dict[int, Tuple[float, float]] = {}
        self.prev_detections: List[Dict] = []
        
        # Статистика
        self.stats = {
            'raw_detections': 0,
            'after_merge': 0,
            'confirmed_objects': 0
        }
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(n):
            hue = (i * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            colors.append(tuple(int(c * 255) for c in rgb[::-1]))
        return colors
    
    def process_frame(self, frame: np.ndarray) -> List[TrackedObject]:
        """Обрабатывает кадр"""
        h, w = frame.shape[:2]
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 1: Background Subtraction
        # ═══════════════════════════════════════════════════════════════════════
        
        # Фильтр яркости
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_mask = np.ones_like(gray) * 255
        brightness_mask[gray < self.config.brightness_min] = 0
        brightness_mask[gray > self.config.brightness_max] = 0
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask[fg_mask == 127] = 0
        
        fg_mask = cv2.bitwise_and(fg_mask, brightness_mask)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 2: Морфология (соединяем близкие области)
        # ═══════════════════════════════════════════════════════════════════════
        
        if self.config.merge.use_morphological_merge:
            # Сначала обычная очистка
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small)
            
            # Затем агрессивный close для соединения частей тела
            kernel_large = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.config.merge.morph_close_size, self.config.merge.morph_close_size)
            )
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large)
            
            # Финальная очистка
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 3: Находим контуры
        # ═══════════════════════════════════════════════════════════════════════
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Фильтруем по размеру
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_area or area > self.config.max_area:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Aspect ratio
            aspect = cw / ch if ch > 0 else 0
            if aspect < self.config.min_aspect_ratio or aspect > self.config.max_aspect_ratio:
                continue
            
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
        
        self.stats['raw_detections'] = len(detections)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 4: Вычисляем скорости детекций (для merge)
        # ═══════════════════════════════════════════════════════════════════════
        
        detection_velocities = self._estimate_detection_velocities(detections)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 5: MERGE детекций
        # ═══════════════════════════════════════════════════════════════════════
        
        if self.config.merge.merge_enabled and len(detections) > 1:
            detections = self.merger.merge_detections(detections, detection_velocities)
        
        # Опционально: кластеризация
        if self.clusterer and len(detections) > 1:
            detections = self.clusterer.cluster(detections, detection_velocities)
        
        self.stats['after_merge'] = len(detections)
        
        # Сохраняем для следующего кадра
        self.prev_detections = detections
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 6: Трекинг
        # ═══════════════════════════════════════════════════════════════════════
        
        for obj in self.objects.values():
            obj.frames_lost += 1
        
        self._match_detections(detections)
        
        # Подтверждаем объекты
        for obj in self.objects.values():
            if not obj.confirmed and obj.frames_seen >= self.config.confirm_frames:
                obj.confirmed = True
                self.stats['confirmed_objects'] += 1
        
        # Удаляем потерянные
        self._cleanup_lost()
        
        return [obj for obj in self.objects.values() if obj.confirmed]
    
    def _estimate_detection_velocities(self, detections: List[Dict]) -> Dict[int, Tuple[float, float]]:
        """Оценивает скорости детекций по сопоставлению с предыдущим кадром"""
        velocities = {}
        
        if not self.prev_detections:
            return velocities
        
        # Простое сопоставление по ближайшему соседу
        for i, det in enumerate(detections):
            min_dist = float('inf')
            best_vel = (0, 0)
            
            for prev_det in self.prev_detections:
                dist = np.sqrt(
                    (det['centroid'][0] - prev_det['centroid'][0])**2 +
                    (det['centroid'][1] - prev_det['centroid'][1])**2
                )
                if dist < min_dist and dist < 150:
                    min_dist = dist
                    best_vel = (
                        det['centroid'][0] - prev_det['centroid'][0],
                        det['centroid'][1] - prev_det['centroid'][1]
                    )
            
            if min_dist < 150:
                velocities[i] = best_vel
        
        return velocities
    
    def _match_detections(self, detections: List[Dict]):
        """Сопоставляет детекции с объектами"""
        if not detections:
            return
        
        used = set()
        
        for obj in list(self.objects.values()):
            if obj.centroid is None:
                continue
            
            predicted = obj.predict_position() or obj.centroid
            
            min_dist = float('inf')
            best_idx = -1
            
            for i, det in enumerate(detections):
                if i in used:
                    continue
                
                dist = min(
                    np.sqrt((obj.centroid[0] - det['centroid'][0])**2 +
                           (obj.centroid[1] - det['centroid'][1])**2),
                    np.sqrt((predicted[0] - det['centroid'][0])**2 +
                           (predicted[1] - det['centroid'][1])**2)
                )
                
                # Velocity check
                if dist > self.config.max_velocity * 1.5:
                    continue
                
                if dist < min_dist and dist < 120:
                    min_dist = dist
                    best_idx = i
            
            if best_idx >= 0:
                det = detections[best_idx]
                obj.update(
                    det['centroid'], 
                    det['bbox'], 
                    det['mask'],
                    det.get('merged_from', 1)
                )
                used.add(best_idx)
        
        # Новые объекты
        for i, det in enumerate(detections):
            if i not in used:
                self._create_object(det)
    
    def _create_object(self, detection: Dict):
        """Создаёт новый объект"""
        obj = TrackedObject(
            id=self.next_id,
            color=self.colors[self.next_id % len(self.colors)]
        )
        obj.update(
            detection['centroid'], 
            detection['bbox'], 
            detection['mask'],
            detection.get('merged_from', 1)
        )
        
        if self.config.use_kalman:
            obj.init_kalman()
        
        self.objects[self.next_id] = obj
        self.next_id += 1
    
    def _cleanup_lost(self):
        """Удаляет потерянные объекты"""
        to_remove = [
            oid for oid, obj in self.objects.items()
            if obj.frames_lost > self.config.lost_frames_max or
               (not obj.confirmed and obj.frames_lost > 3)
        ]
        for oid in to_remove:
            del self.objects[oid]


# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """Визуализация"""
    
    def __init__(self, show_debug: bool = False):
        self.show_debug = show_debug
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def render(self, frame: np.ndarray, objects: List[TrackedObject],
               tracker: AdvancedTrackerV2, frame_idx: int,
               total_frames: int, fps: float) -> np.ndarray:
        
        output = frame.copy()
        
        # Маски
        for obj in objects:
            if obj.mask is not None:
                overlay = output.copy()
                overlay[obj.mask > 0] = obj.color
                cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)
                
                contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, contours, -1, obj.color, 2)
        
        # Траектории
        for obj in objects:
            if len(obj.trajectory) > 1:
                pts = list(obj.trajectory)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    color = tuple(int(c * alpha) for c in obj.color)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(output,
                            tuple(map(int, pts[i-1])),
                            tuple(map(int, pts[i])),
                            color, thickness)
        
        # Bboxes
        for obj in objects:
            if obj.bbox:
                x, y, w, h = obj.bbox
                
                corner = min(20, w // 5, h // 5)
                corners = [
                    [(x, y), (x + corner, y)],
                    [(x, y), (x, y + corner)],
                    [(x + w, y), (x + w - corner, y)],
                    [(x + w, y), (x + w, y + corner)],
                    [(x, y + h), (x + corner, y + h)],
                    [(x, y + h), (x, y + h - corner)],
                    [(x + w, y + h), (x + w - corner, y + h)],
                    [(x + w, y + h), (x + w, y + h - corner)],
                ]
                for p1, p2 in corners:
                    cv2.line(output, p1, p2, obj.color, 2)
                
                # Метка
                label = f"#{obj.id}"
                if obj.merged_count > 1:
                    label += f" [M:{obj.merged_count}]"
                
                speed = obj.get_speed()
                if speed > 1:
                    label += f" v:{speed:.0f}"
                
                cv2.putText(output, label, (x, y - 8),
                           self.font, 0.5, (0, 0, 0), 3)
                cv2.putText(output, label, (x, y - 8),
                           self.font, 0.5, obj.color, 1)
        
        # Центроиды
        for obj in objects:
            if obj.smoothed_pos:
                center = tuple(map(int, obj.smoothed_pos))
                cv2.circle(output, center, 6, (255, 255, 255), 2)
                cv2.circle(output, center, 4, obj.color, -1)
        
        # Инфо-панель
        self._draw_info(output, objects, tracker, frame_idx, total_frames, fps)
        
        return output
    
    def _draw_info(self, frame: np.ndarray, objects: List[TrackedObject],
                   tracker: AdvancedTrackerV2, frame_idx: int,
                   total_frames: int, fps: float):
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 35
        cv2.putText(frame, "TRACKER v2 + MERGE", (20, y),
                   self.font, 0.7, (100, 200, 255), 2)
        
        y += 25
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames} | FPS: {fps:.1f}", 
                   (20, y), self.font, 0.45, (200, 200, 200), 1)
        
        y += 22
        cv2.putText(frame, f"Raw detections: {tracker.stats['raw_detections']}", 
                   (20, y), self.font, 0.4, (200, 200, 200), 1)
        
        y += 18
        cv2.putText(frame, f"After merge: {tracker.stats['after_merge']}", 
                   (20, y), self.font, 0.4, (100, 255, 100), 1)
        
        y += 18
        cv2.putText(frame, f"Confirmed objects: {len(objects)}", 
                   (20, y), self.font, 0.4, (255, 200, 100), 1)
        
        y += 22
        progress = (frame_idx + 1) / total_frames
        bar_w = 280
        cv2.rectangle(frame, (20, y), (20 + bar_w, y + 8), (50, 50, 60), -1)
        cv2.rectangle(frame, (20, y), (20 + int(bar_w * progress), y + 8),
                     (100, 200, 255), -1)


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 ПРОЦЕССОР
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessorV2:
    """Процессор видео"""
    
    def __init__(self, config: Optional[TrackerConfig] = None, show_debug: bool = False):
        self.tracker = AdvancedTrackerV2(config)
        self.visualizer = Visualizer(show_debug)
    
    def process_video(self, input_path: str, output_path: str,
                      show_preview: bool = True,
                      max_frames: Optional[int] = None):
        
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
        
        merge_cfg = self.tracker.config.merge
        
        print(f"\n{'═'*65}")
        print(f"  🎭 ADVANCED TRACKER v2 + MERGE")
        print(f"{'═'*65}")
        print(f"  📁 Input:  {input_path}")
        print(f"  📁 Output: {output_path}")
        print(f"  📐 Size:   {width}x{height} @ {fps:.1f} FPS")
        print(f"{'─'*65}")
        print(f"  🔗 MERGE CONFIG:")
        print(f"     • Distance merge: {merge_cfg.merge_distance} px")
        print(f"     • Velocity merge: {'ON' if merge_cfg.velocity_merge_enabled else 'OFF'} (threshold: {merge_cfg.velocity_similarity_threshold})")
        print(f"     • IoU merge: {'ON' if merge_cfg.iou_merge_enabled else 'OFF'} (threshold: {merge_cfg.iou_threshold})")
        print(f"     • Vertical merge: {'ON' if merge_cfg.vertical_merge_enabled else 'OFF'}")
        print(f"     • Morph close: {merge_cfg.morph_close_size}px")
        print(f"{'═'*65}")
        print(f"\n  Press Q to quit, P to pause\n")
        
        frame_times = []
        frame_idx = 0
        
        while frame_idx < total:
            ret, frame = cap.read()
            if not ret:
                break
            
            start = time.time()
            
            objects = self.tracker.process_frame(frame)
            
            current_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else fps
            output_frame = self.visualizer.render(
                frame, objects, self.tracker, frame_idx, total, current_fps
            )
            
            frame_time = time.time() - start
            frame_times.append(frame_time)
            
            out.write(output_frame)
            
            if show_preview:
                preview = cv2.resize(output_frame, (0, 0), fx=0.6, fy=0.6)
                cv2.imshow('Tracker v2 + Merge', preview)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  Interrupted")
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
            
            if frame_idx % 30 == 0:
                progress = (frame_idx + 1) / total * 100
                avg_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
                raw = self.tracker.stats['raw_detections']
                merged = self.tracker.stats['after_merge']
                print(f"\r  [{progress:5.1f}%] FPS: {avg_fps:.1f} | Raw: {raw} → Merged: {merged} | Objects: {len(objects)}", end="")
            
            frame_idx += 1
        
        print(f"\n\n{'═'*65}")
        print(f"  ✅ COMPLETE")
        print(f"{'═'*65}")
        if frame_times:
            print(f"  ⏱  Avg time: {np.mean(frame_times)*1000:.1f} ms")
            print(f"  🚀 Avg FPS: {1.0/np.mean(frame_times):.1f}")
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
        description='🎭 Advanced Tracker v2 + MERGE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python advanced_tracker_v2.py test.mp4
  python advanced_tracker_v2.py test.mp4 --merge-distance 100
  python advanced_tracker_v2.py test.mp4 --velocity-threshold 0.9
  python advanced_tracker_v2.py test.mp4 --morph-close 20
        """
    )
    
    parser.add_argument('input', help='Input video')
    parser.add_argument('-o', '--output', default=None)
    
    # Merge parameters
    parser.add_argument('--merge-distance', type=float, default=80.0,
                       help='Max distance for merge (px)')
    parser.add_argument('--velocity-threshold', type=float, default=0.7,
                       help='Velocity similarity threshold (0-1)')
    parser.add_argument('--iou-threshold', type=float, default=0.1,
                       help='IoU threshold for merge')
    parser.add_argument('--morph-close', type=int, default=15,
                       help='Morphological close kernel size')
    
    # Disable features
    parser.add_argument('--no-velocity-merge', action='store_true')
    parser.add_argument('--no-iou-merge', action='store_true')
    parser.add_argument('--no-vertical-merge', action='store_true')
    parser.add_argument('--no-clustering', action='store_true')
    
    # General
    parser.add_argument('--min-area', type=int, default=1500)
    parser.add_argument('--confirm-frames', type=int, default=3)
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--max-frames', type=int, default=None)
    
    args = parser.parse_args()
    
    # Config
    merge_config = MergeConfig(
        merge_distance=args.merge_distance,
        velocity_similarity_threshold=args.velocity_threshold,
        iou_threshold=args.iou_threshold,
        morph_close_size=args.morph_close,
        velocity_merge_enabled=not args.no_velocity_merge,
        iou_merge_enabled=not args.no_iou_merge,
        vertical_merge_enabled=not args.no_vertical_merge,
        use_clustering=not args.no_clustering
    )
    
    config = TrackerConfig(
        min_area=args.min_area,
        confirm_frames=args.confirm_frames,
        merge=merge_config
    )
    
    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_merged.mp4")
    
    processor = ProcessorV2(config)
    processor.process_video(
        args.input,
        args.output,
        show_preview=not args.no_preview,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()


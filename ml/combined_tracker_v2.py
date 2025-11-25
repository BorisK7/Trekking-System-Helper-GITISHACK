#!/usr/bin/env python3
"""
🎭 Combined Tracker v2 - С РЕТРОСПЕКТИВНЫМ ФИЛЬТРОМ
====================================================
Улучшения:
1. Ретроспективный фильтр: объект показывается только если он "выжил" N кадров
2. Более строгий фильтр по скорости
3. Merge близких сегментов в один объект

Запуск:
    python combined_tracker_v2.py test.mp4
    python combined_tracker_v2.py test.mp4 --min-lifetime 10
    python combined_tracker_v2.py test.mp4 --merge-distance 60
"""

import cv2
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import colorsys
import time


# ═══════════════════════════════════════════════════════════════════════════════
# 🦴 СКЕЛЕТ
# ═══════════════════════════════════════════════════════════════════════════════

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

LIMB_COLORS = {
    'head': (255, 200, 100),
    'torso': (100, 255, 100),
    'left_arm': (255, 100, 100),
    'right_arm': (100, 100, 255),
    'left_leg': (255, 100, 255),
    'right_leg': (100, 255, 255),
}


def get_limb_color(i: int, j: int) -> Tuple[int, int, int]:
    if i <= 4 or j <= 4:
        return LIMB_COLORS['head']
    elif (i in [5, 6, 11, 12]) and (j in [5, 6, 11, 12]):
        return LIMB_COLORS['torso']
    elif i in [5, 7, 9] or j in [5, 7, 9]:
        return LIMB_COLORS['left_arm']
    elif i in [6, 8, 10] or j in [6, 8, 10]:
        return LIMB_COLORS['right_arm']
    elif i in [11, 13, 15] or j in [11, 13, 15]:
        return LIMB_COLORS['left_leg']
    return LIMB_COLORS['right_leg']


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Конфигурация"""
    
    # YOLO
    yolo_model: str = 'n'
    yolo_confidence: float = 0.3
    
    # Сегментация
    min_area: int = 2000              # Увеличено
    max_area: int = 120000
    
    # === ФИЛЬТР ПО СКОРОСТИ (более строгий) ===
    min_speed: float = 4.0            # Увеличено с 2.0
    speed_window: int = 15            # Увеличено окно
    min_moving_ratio: float = 0.5     # Минимум 50% кадров должен двигаться
    
    # === РЕТРОСПЕКТИВНЫЙ ФИЛЬТР ===
    min_lifetime: int = 8             # Объект должен "выжить" минимум N кадров
    retrospective_buffer: int = 30    # Буфер для ретроспективного анализа
    
    # === MERGE СЕГМЕНТОВ ===
    merge_enabled: bool = True
    merge_distance: float = 60.0      # Расстояние для merge центроидов
    merge_iou_threshold: float = 0.05 # IoU для merge bbox'ов
    
    # Temporal
    confirm_frames: int = 5           # Увеличено
    lost_frames_max: int = 10         # Уменьшено - быстрее удаляем


# ═══════════════════════════════════════════════════════════════════════════════
# 🔗 MERGE СЕГМЕНТОВ
# ═══════════════════════════════════════════════════════════════════════════════

class SegmentMerger:
    """Объединяет близкие сегменты"""
    
    def __init__(self, distance_threshold: float = 60.0, iou_threshold: float = 0.05):
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
    
    def merge(self, detections: List[Dict]) -> List[Dict]:
        """Объединяет близкие детекции"""
        if len(detections) < 2:
            return detections
        
        n = len(detections)
        
        # Строим граф связей
        merge_graph = [[False] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(detections[i], detections[j]):
                    merge_graph[i][j] = True
                    merge_graph[j][i] = True
        
        # Находим компоненты связности
        visited = [False] * n
        clusters = []
        
        def dfs(node: int, cluster: List[int]):
            visited[node] = True
            cluster.append(node)
            for neighbor in range(n):
                if merge_graph[node][neighbor] and not visited[neighbor]:
                    dfs(neighbor, cluster)
        
        for i in range(n):
            if not visited[i]:
                cluster = []
                dfs(i, cluster)
                clusters.append(cluster)
        
        # Объединяем детекции в каждом кластере
        merged = []
        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(detections[cluster[0]])
            else:
                merged.append(self._merge_cluster([detections[i] for i in cluster]))
        
        return merged
    
    def _should_merge(self, det1: Dict, det2: Dict) -> bool:
        """Определяет, нужно ли объединять"""
        c1, c2 = det1['centroid'], det2['centroid']
        
        # По расстоянию
        distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        if distance < self.distance_threshold:
            return True
        
        # По IoU
        iou = self._compute_iou(det1['bbox'], det2['bbox'])
        if iou > self.iou_threshold:
            return True
        
        # По вертикали (части одного объекта)
        if self._is_vertically_adjacent(det1['bbox'], det2['bbox']):
            return True
        
        return False
    
    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_vertically_adjacent(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """Проверяет вертикальную смежность"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Определяем верхний и нижний
        if y1 < y2:
            upper, lower = bbox1, bbox2
        else:
            upper, lower = bbox2, bbox1
        
        ux, uy, uw, uh = upper
        lx, ly, lw, lh = lower
        
        # Вертикальный зазор
        gap = ly - (uy + uh)
        if gap < 0 or gap > 40:
            return False
        
        # Горизонтальное перекрытие
        overlap_left = max(ux, lx)
        overlap_right = min(ux + uw, lx + lw)
        
        if overlap_right <= overlap_left:
            return False
        
        overlap_ratio = (overlap_right - overlap_left) / min(uw, lw)
        return overlap_ratio > 0.4
    
    def _merge_cluster(self, detections: List[Dict]) -> Dict:
        """Объединяет кластер детекций"""
        # Объединяем bbox
        x_min = min(d['bbox'][0] for d in detections)
        y_min = min(d['bbox'][1] for d in detections)
        x_max = max(d['bbox'][0] + d['bbox'][2] for d in detections)
        y_max = max(d['bbox'][1] + d['bbox'][3] for d in detections)
        
        merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Центроид по площади
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
            'merged_count': len(detections)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 TRACKED PERSON
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedPerson:
    id: int
    color: Tuple[int, int, int]
    
    keypoints: Optional[np.ndarray] = None
    confidences: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    
    trajectory: deque = field(default_factory=lambda: deque(maxlen=60))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=20))
    
    frames_seen: int = 0
    frames_lost: int = 0
    confirmed: bool = False
    
    # Ретроспективный буфер
    history_buffer: deque = field(default_factory=lambda: deque(maxlen=30))
    
    def get_center(self) -> Optional[Tuple[float, float]]:
        if self.keypoints is not None:
            left_hip = self.keypoints[11]
            right_hip = self.keypoints[12]
            if left_hip[0] > 0 and right_hip[0] > 0:
                return ((left_hip[0] + right_hip[0]) / 2,
                       (left_hip[1] + right_hip[1]) / 2)
        if self.bbox:
            x, y, w, h = self.bbox
            return (x + w / 2, y + h / 2)
        return None
    
    def update(self, keypoints: np.ndarray, confidences: np.ndarray,
               bbox: Tuple[int, int, int, int]):
        prev_center = self.get_center()
        
        self.keypoints = keypoints
        self.confidences = confidences
        self.bbox = bbox
        
        center = self.get_center()
        if center:
            self.trajectory.append(center)
            
            if prev_center:
                speed = np.sqrt((center[0] - prev_center[0])**2 +
                               (center[1] - prev_center[1])**2)
                self.speed_history.append(speed)
        
        # Сохраняем в буфер
        self.history_buffer.append({
            'keypoints': keypoints.copy(),
            'bbox': bbox,
            'center': center
        })
        
        self.frames_seen += 1
        self.frames_lost = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 TRACKED OBJECT (с ретроспективным фильтром)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedObject:
    id: int
    color: Tuple[int, int, int]
    
    centroid: Optional[Tuple[float, float]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    mask: Optional[np.ndarray] = None
    
    trajectory: deque = field(default_factory=lambda: deque(maxlen=60))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=30))
    
    frames_seen: int = 0
    frames_lost: int = 0
    total_lifetime: int = 0           # Общее время жизни
    
    # Ретроспективный буфер - храним историю для отрисовки после подтверждения
    history_buffer: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Статусы
    is_moving: bool = False
    is_valid: bool = False            # Прошёл ретроспективный фильтр
    moving_frame_count: int = 0       # Кадры с реальным движением
    
    def get_average_speed(self) -> float:
        if len(self.speed_history) < 3:
            return 0
        # Берём медиану вместо среднего (устойчивее к выбросам)
        return float(np.median(list(self.speed_history)))
    
    def get_moving_ratio(self) -> float:
        """Доля кадров, когда объект двигался"""
        if self.frames_seen < 3:
            return 0
        return self.moving_frame_count / self.frames_seen
    
    def update(self, centroid: Tuple[float, float], bbox: Tuple[int, int, int, int],
               mask: np.ndarray, min_speed: float = 4.0):
        prev_centroid = self.centroid
        
        self.centroid = centroid
        self.bbox = bbox
        self.mask = mask
        self.trajectory.append(centroid)
        
        # Скорость
        speed = 0
        if prev_centroid:
            speed = np.sqrt((centroid[0] - prev_centroid[0])**2 +
                           (centroid[1] - prev_centroid[1])**2)
            self.speed_history.append(speed)
            
            if speed > min_speed:
                self.moving_frame_count += 1
        
        # Сохраняем в буфер
        self.history_buffer.append({
            'centroid': centroid,
            'bbox': bbox,
            'mask': mask.copy() if mask is not None else None,
            'speed': speed
        })
        
        self.frames_seen += 1
        self.total_lifetime += 1
        self.frames_lost = 0
    
    def mark_lost(self):
        """Помечает объект как потерянный"""
        self.frames_lost += 1
        self.total_lifetime += 1


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 КОМБИНИРОВАННЫЙ ТРЕКЕР v2
# ═══════════════════════════════════════════════════════════════════════════════

class CombinedTrackerV2:
    """
    Комбинированный трекер v2:
    - YOLO для людей
    - Сегментация + merge + фильтр скорости + ретроспективный фильтр
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        self.yolo_model = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        
        self.segment_merger = SegmentMerger(
            config.merge_distance, config.merge_iou_threshold
        ) if config.merge_enabled else None
        
        self.persons: Dict[int, TrackedPerson] = {}
        self.objects: Dict[int, TrackedObject] = {}
        self.next_person_id = 0
        self.next_object_id = 0
        
        self.person_colors = self._generate_colors(20, 0.9)
        self.object_colors = self._generate_colors(20, 0.7, 0.5)
        
        # Статистика
        self.stats = {
            'raw_segments': 0,
            'after_merge': 0,
            'moving': 0,
            'valid': 0,  # Прошли ретроспективный фильтр
            'persons': 0
        }
    
    def _generate_colors(self, n: int, sat: float = 0.9, offset: float = 0.0):
        colors = []
        for i in range(n):
            hue = ((i * 0.618033988749895) + offset) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, sat, 1.0)
            colors.append(tuple(int(c * 255) for c in rgb[::-1]))
        return colors
    
    def _load_yolo(self):
        if self.yolo_model is not None:
            return
        from ultralytics import YOLO
        self.yolo_model = YOLO(f'yolov8{self.config.yolo_model}-pose.pt')
        print(f"✓ YOLOv8{self.config.yolo_model}-pose loaded")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[TrackedPerson], List[TrackedObject]]:
        h, w = frame.shape[:2]
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 1: YOLO
        # ═══════════════════════════════════════════════════════════════════════
        
        self._load_yolo()
        
        yolo_results = self.yolo_model(frame, verbose=False, conf=self.config.yolo_confidence)
        
        person_detections = []
        person_masks = np.zeros((h, w), dtype=np.uint8)
        
        for result in yolo_results:
            if result.keypoints is None:
                continue
            
            keypoints = result.keypoints.xy.cpu().numpy()
            confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
            
            for i, kpts in enumerate(keypoints):
                conf = confidences[i] if confidences is not None else np.ones(17)
                
                if boxes is not None and i < len(boxes):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    # Расширяем маску для исключения
                    pad = 20
                    cv2.rectangle(person_masks, 
                                 (max(0, x1 - pad), max(0, y1 - pad)),
                                 (min(w, x2 + pad), min(h, y2 + pad)), 255, -1)
                else:
                    valid_pts = kpts[kpts[:, 0] > 0]
                    if len(valid_pts) == 0:
                        continue
                    x1, y1 = valid_pts.min(axis=0).astype(int)
                    x2, y2 = valid_pts.max(axis=0).astype(int)
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                
                person_detections.append({
                    'keypoints': kpts,
                    'confidences': conf,
                    'bbox': bbox
                })
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 2: Background Subtraction
        # ═══════════════════════════════════════════════════════════════════════
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_mask = np.ones_like(gray) * 255
        brightness_mask[gray < 20] = 0
        brightness_mask[gray > 240] = 0
        
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask[fg_mask == 127] = 0
        
        fg_mask = cv2.bitwise_and(fg_mask, brightness_mask)
        fg_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(person_masks))
        
        # Морфология
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        object_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_area or area > self.config.max_area:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + cw / 2, y + ch / 2
            
            obj_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(obj_mask, [contour], -1, 255, -1)
            
            object_detections.append({
                'centroid': (cx, cy),
                'bbox': (x, y, cw, ch),
                'mask': obj_mask,
                'area': area
            })
        
        self.stats['raw_segments'] = len(object_detections)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 3: MERGE сегментов
        # ═══════════════════════════════════════════════════════════════════════
        
        if self.segment_merger and len(object_detections) > 1:
            object_detections = self.segment_merger.merge(object_detections)
        
        self.stats['after_merge'] = len(object_detections)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 4: Трекинг людей
        # ═══════════════════════════════════════════════════════════════════════
        
        for person in self.persons.values():
            person.frames_lost += 1
        
        self._match_persons(person_detections)
        
        for person in self.persons.values():
            if not person.confirmed and person.frames_seen >= self.config.confirm_frames:
                person.confirmed = True
        
        to_remove = [pid for pid, p in self.persons.items()
                    if p.frames_lost > self.config.lost_frames_max]
        for pid in to_remove:
            del self.persons[pid]
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 5: Трекинг объектов
        # ═══════════════════════════════════════════════════════════════════════
        
        for obj in self.objects.values():
            obj.mark_lost()
        
        self._match_objects(object_detections)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 6: Фильтр по скорости + ретроспективный
        # ═══════════════════════════════════════════════════════════════════════
        
        for obj in self.objects.values():
            avg_speed = obj.get_average_speed()
            moving_ratio = obj.get_moving_ratio()
            
            # Объект движется если:
            # - Средняя скорость выше порога
            # - И он двигался хотя бы min_moving_ratio кадров
            obj.is_moving = (
                avg_speed > self.config.min_speed and
                moving_ratio > self.config.min_moving_ratio
            )
            
            # РЕТРОСПЕКТИВНЫЙ ФИЛЬТР:
            # Объект валиден если он "выжил" достаточно долго
            obj.is_valid = (
                obj.is_moving and
                obj.total_lifetime >= self.config.min_lifetime and
                obj.frames_seen >= self.config.min_lifetime // 2
            )
        
        # Очистка потерянных
        to_remove = [oid for oid, o in self.objects.items()
                    if o.frames_lost > self.config.lost_frames_max]
        for oid in to_remove:
            del self.objects[oid]
        
        # ═══════════════════════════════════════════════════════════════════════
        # Результаты
        # ═══════════════════════════════════════════════════════════════════════
        
        confirmed_persons = [p for p in self.persons.values() if p.confirmed]
        
        # Только валидные объекты (прошли ретроспективный фильтр)
        valid_objects = [o for o in self.objects.values() if o.is_valid]
        
        self.stats['persons'] = len(confirmed_persons)
        self.stats['moving'] = len([o for o in self.objects.values() if o.is_moving])
        self.stats['valid'] = len(valid_objects)
        
        return confirmed_persons, valid_objects
    
    def _match_persons(self, detections: List[Dict]):
        if not detections:
            return
        
        used = set()
        
        for person in list(self.persons.values()):
            center = person.get_center()
            if center is None:
                continue
            
            min_dist = float('inf')
            best_idx = -1
            
            for i, det in enumerate(detections):
                if i in used:
                    continue
                
                x, y, w, h = det['bbox']
                det_center = (x + w / 2, y + h / 2)
                
                dist = np.sqrt((center[0] - det_center[0])**2 +
                              (center[1] - det_center[1])**2)
                
                if dist < min_dist and dist < 150:
                    min_dist = dist
                    best_idx = i
            
            if best_idx >= 0:
                det = detections[best_idx]
                person.update(det['keypoints'], det['confidences'], det['bbox'])
                used.add(best_idx)
        
        for i, det in enumerate(detections):
            if i not in used:
                person = TrackedPerson(
                    id=self.next_person_id,
                    color=self.person_colors[self.next_person_id % len(self.person_colors)]
                )
                person.update(det['keypoints'], det['confidences'], det['bbox'])
                self.persons[self.next_person_id] = person
                self.next_person_id += 1
    
    def _match_objects(self, detections: List[Dict]):
        if not detections:
            return
        
        used = set()
        
        for obj in list(self.objects.values()):
            if obj.centroid is None:
                continue
            
            min_dist = float('inf')
            best_idx = -1
            
            for i, det in enumerate(detections):
                if i in used:
                    continue
                
                dist = np.sqrt((obj.centroid[0] - det['centroid'][0])**2 +
                              (obj.centroid[1] - det['centroid'][1])**2)
                
                if dist < min_dist and dist < 80:
                    min_dist = dist
                    best_idx = i
            
            if best_idx >= 0:
                det = detections[best_idx]
                obj.update(det['centroid'], det['bbox'], det['mask'],
                          self.config.min_speed)
                used.add(best_idx)
        
        for i, det in enumerate(detections):
            if i not in used:
                obj = TrackedObject(
                    id=self.next_object_id,
                    color=self.object_colors[self.next_object_id % len(self.object_colors)]
                )
                obj.update(det['centroid'], det['bbox'], det['mask'],
                          self.config.min_speed)
                self.objects[self.next_object_id] = obj
                self.next_object_id += 1


# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_skeleton(self, frame: np.ndarray, person: TrackedPerson, min_conf: float = 0.3):
        if person.keypoints is None:
            return frame
        
        kpts = person.keypoints
        confs = person.confidences if person.confidences is not None else np.ones(17)
        
        for i, j in SKELETON_CONNECTIONS:
            if i >= len(kpts) or j >= len(kpts):
                continue
            pt1, pt2 = kpts[i], kpts[j]
            if pt1[0] <= 0 or pt2[0] <= 0 or confs[i] < min_conf or confs[j] < min_conf:
                continue
            
            pt1, pt2 = tuple(map(int, pt1)), tuple(map(int, pt2))
            color = get_limb_color(i, j)
            
            cv2.line(frame, pt1, pt2, tuple(c // 4 for c in color), 8)
            cv2.line(frame, pt1, pt2, tuple(c // 2 for c in color), 5)
            cv2.line(frame, pt1, pt2, color, 3)
        
        for i, pt in enumerate(kpts):
            if pt[0] <= 0 or confs[i] < min_conf:
                continue
            pt = tuple(map(int, pt))
            cv2.circle(frame, pt, 6, person.color, -1)
            cv2.circle(frame, pt, 3, (255, 255, 255), -1)
        
        return frame
    
    def draw_object(self, frame: np.ndarray, obj: TrackedObject):
        if obj.mask is None:
            return frame
        
        color = obj.color
        
        overlay = frame.copy()
        overlay[obj.mask > 0] = color
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 2)
        
        if obj.bbox:
            x, y, w, h = obj.bbox
            speed = obj.get_average_speed()
            label = f"Obj #{obj.id} v:{speed:.1f}"
            cv2.putText(frame, label, (x, y - 8), self.font, 0.45, (0, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 8), self.font, 0.45, color, 1)
        
        return frame
    
    def draw_trajectory(self, frame: np.ndarray, trajectory: deque, color: Tuple):
        if len(trajectory) < 2:
            return frame
        pts = list(trajectory)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            pt_color = tuple(int(c * alpha) for c in color)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, tuple(map(int, pts[i-1])), tuple(map(int, pts[i])), pt_color, thickness)
        return frame
    
    def draw_person_info(self, frame: np.ndarray, person: TrackedPerson):
        if person.bbox is None:
            return frame
        x, y, w, h = person.bbox
        color = person.color
        
        corner = min(15, w // 5, h // 5)
        corners = [
            [(x, y), (x + corner, y)], [(x, y), (x, y + corner)],
            [(x + w, y), (x + w - corner, y)], [(x + w, y), (x + w, y + corner)],
            [(x, y + h), (x + corner, y + h)], [(x, y + h), (x, y + h - corner)],
            [(x + w, y + h), (x + w - corner, y + h)], [(x + w, y + h), (x + w, y + h - corner)],
        ]
        for p1, p2 in corners:
            cv2.line(frame, p1, p2, color, 2)
        
        label = f"Person #{person.id}"
        cv2.putText(frame, label, (x, y - 8), self.font, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, label, (x, y - 8), self.font, 0.5, color, 1)
        return frame
    
    def render(self, frame: np.ndarray, persons: List[TrackedPerson], objects: List[TrackedObject]):
        output = frame.copy()
        
        for person in persons:
            output = self.draw_trajectory(output, person.trajectory, person.color)
        for obj in objects:
            output = self.draw_trajectory(output, obj.trajectory, obj.color)
        
        for obj in objects:
            output = self.draw_object(output, obj)
        
        for person in persons:
            output = self.draw_skeleton(output, person)
            output = self.draw_person_info(output, person)
        
        return output
    
    def draw_info(self, frame: np.ndarray, tracker: CombinedTrackerV2,
                  frame_idx: int, total_frames: int, fps: float):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (340, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 35
        cv2.putText(frame, "COMBINED TRACKER v2", (20, y), self.font, 0.65, (100, 200, 255), 2)
        
        y += 22
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames} | FPS: {fps:.1f}",
                   (20, y), self.font, 0.4, (200, 200, 200), 1)
        
        y += 20
        cv2.putText(frame, f"People (YOLO): {tracker.stats['persons']}",
                   (20, y), self.font, 0.4, (100, 255, 100), 1)
        
        y += 18
        cv2.putText(frame, f"Segments: {tracker.stats['raw_segments']} -> merge: {tracker.stats['after_merge']}",
                   (20, y), self.font, 0.35, (150, 150, 150), 1)
        
        y += 18
        cv2.putText(frame, f"Moving: {tracker.stats['moving']} -> valid: {tracker.stats['valid']}",
                   (20, y), self.font, 0.4, (255, 200, 100), 1)
        
        y += 18
        cv2.putText(frame, f"Min speed: {tracker.config.min_speed} | Min lifetime: {tracker.config.min_lifetime}",
                   (20, y), self.font, 0.3, (120, 120, 120), 1)
        
        y += 12
        progress = (frame_idx + 1) / total_frames
        bar_w = 300
        cv2.rectangle(frame, (20, y), (20 + bar_w, y + 6), (50, 50, 60), -1)
        cv2.rectangle(frame, (20, y), (20 + int(bar_w * progress), y + 6), (100, 200, 255), -1)
        
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 ПРОЦЕССОР
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessorV2:
    def __init__(self, config: Config):
        self.tracker = CombinedTrackerV2(config)
        self.visualizer = Visualizer()
        self.config = config
    
    def process_video(self, input_path: str, output_path: str,
                      show_preview: bool = True, max_frames: Optional[int] = None):
        
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
        
        print(f"\n{'═'*65}")
        print(f"  🎭 COMBINED TRACKER v2 + RETROSPECTIVE FILTER")
        print(f"{'═'*65}")
        print(f"  📁 Input:  {input_path}")
        print(f"  📁 Output: {output_path}")
        print(f"{'─'*65}")
        print(f"  ⚙️  Min speed: {self.config.min_speed} px/frame")
        print(f"  ⚙️  Min moving ratio: {self.config.min_moving_ratio}")
        print(f"  ⚙️  Min lifetime: {self.config.min_lifetime} frames")
        print(f"  ⚙️  Merge distance: {self.config.merge_distance} px")
        print(f"{'═'*65}")
        print(f"\n  Press Q to quit, P to pause\n")
        
        frame_times = []
        frame_idx = 0
        
        while frame_idx < total:
            ret, frame = cap.read()
            if not ret:
                break
            
            start = time.time()
            
            persons, objects = self.tracker.process_frame(frame)
            
            output_frame = self.visualizer.render(frame, persons, objects)
            
            current_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else fps
            output_frame = self.visualizer.draw_info(
                output_frame, self.tracker, frame_idx, total, current_fps
            )
            
            frame_time = time.time() - start
            frame_times.append(frame_time)
            
            out.write(output_frame)
            
            if show_preview:
                preview = cv2.resize(output_frame, (0, 0), fx=0.6, fy=0.6)
                cv2.imshow('Combined Tracker v2', preview)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  Interrupted")
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
            
            if frame_idx % 30 == 0:
                progress = (frame_idx + 1) / total * 100
                avg_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
                s = self.tracker.stats
                print(f"\r  [{progress:5.1f}%] FPS: {avg_fps:.1f} | People: {s['persons']} | Valid: {s['valid']}/{s['after_merge']}", end="")
            
            frame_idx += 1
        
        print(f"\n\n{'═'*65}")
        print(f"  ✅ COMPLETE - Saved: {output_path}")
        print(f"{'═'*65}\n")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='🎭 Combined Tracker v2')
    
    parser.add_argument('input', help='Input video')
    parser.add_argument('-o', '--output', default=None)
    
    parser.add_argument('--model', '-m', default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--conf', type=float, default=0.3)
    
    # Фильтры
    parser.add_argument('--min-speed', type=float, default=4.0,
                       help='Minimum speed (px/frame)')
    parser.add_argument('--min-moving-ratio', type=float, default=0.5,
                       help='Min ratio of moving frames')
    parser.add_argument('--min-lifetime', type=int, default=8,
                       help='Min frames object must survive')
    
    # Merge
    parser.add_argument('--merge-distance', type=float, default=60.0)
    parser.add_argument('--no-merge', action='store_true')
    
    parser.add_argument('--min-area', type=int, default=2000)
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--max-frames', type=int, default=None)
    
    args = parser.parse_args()
    
    config = Config(
        yolo_model=args.model,
        yolo_confidence=args.conf,
        min_area=args.min_area,
        min_speed=args.min_speed,
        min_moving_ratio=args.min_moving_ratio,
        min_lifetime=args.min_lifetime,
        merge_enabled=not args.no_merge,
        merge_distance=args.merge_distance
    )
    
    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_v2.mp4")
    
    processor = ProcessorV2(config)
    processor.process_video(args.input, args.output,
                           show_preview=not args.no_preview,
                           max_frames=args.max_frames)


if __name__ == '__main__':
    main()


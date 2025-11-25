#!/usr/bin/env python3
"""
🎭 Combined Tracker: YOLO Skeletons + Motion Segmentation
=========================================================
Комбинированный трекер:
- YOLO-pose для скелетов людей
- Сегментация для других движущихся объектов (реквизит, декорации)
- Фильтр по скорости: показываем только реально движущиеся объекты

Запуск:
    python combined_tracker.py test.mp4
    python combined_tracker.py test.mp4 --min-speed 3
    python combined_tracker.py test.mp4 --speed-window 10
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
# 🦴 СКЕЛЕТ (YOLO)
# ═══════════════════════════════════════════════════════════════════════════════

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Голова
    (5, 6), (5, 11), (6, 12), (11, 12),  # Торс
    (5, 7), (7, 9),  # Левая рука
    (6, 8), (8, 10),  # Правая рука
    (11, 13), (13, 15),  # Левая нога
    (12, 14), (14, 16)  # Правая нога
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
    """Конфигурация трекера"""
    
    # YOLO
    yolo_model: str = 'n'
    yolo_confidence: float = 0.3
    
    # Сегментация
    min_area: int = 1500
    max_area: int = 150000
    
    # Фильтр по скорости (КЛЮЧЕВОЙ)
    min_speed: float = 2.0           # Минимальная скорость для показа (px/frame)
    speed_window: int = 10            # Окно для усреднения скорости (кадры)
    speed_threshold_frames: int = 5   # Сколько кадров объект должен двигаться
    
    # Temporal
    confirm_frames: int = 3
    lost_frames_max: int = 15


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 TRACKED PERSON (YOLO)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedPerson:
    """Человек с скелетом"""
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
    
    def get_center(self) -> Optional[Tuple[float, float]]:
        if self.keypoints is not None:
            # Центр бёдер
            left_hip = self.keypoints[11]
            right_hip = self.keypoints[12]
            if left_hip[0] > 0 and right_hip[0] > 0:
                return ((left_hip[0] + right_hip[0]) / 2,
                       (left_hip[1] + right_hip[1]) / 2)
        if self.bbox:
            x, y, w, h = self.bbox
            return (x + w / 2, y + h / 2)
        return None
    
    def get_average_speed(self) -> float:
        if len(self.speed_history) < 2:
            return 0
        return float(np.mean(list(self.speed_history)))
    
    def update(self, keypoints: np.ndarray, confidences: np.ndarray,
               bbox: Tuple[int, int, int, int]):
        prev_center = self.get_center()
        
        self.keypoints = keypoints
        self.confidences = confidences
        self.bbox = bbox
        
        center = self.get_center()
        if center:
            self.trajectory.append(center)
            
            # Вычисляем скорость
            if prev_center:
                speed = np.sqrt(
                    (center[0] - prev_center[0])**2 +
                    (center[1] - prev_center[1])**2
                )
                self.speed_history.append(speed)
        
        self.frames_seen += 1
        self.frames_lost = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 TRACKED OBJECT (СЕГМЕНТАЦИЯ)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedObject:
    """Сегментированный объект"""
    id: int
    color: Tuple[int, int, int]
    
    centroid: Optional[Tuple[float, float]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    mask: Optional[np.ndarray] = None
    
    trajectory: deque = field(default_factory=lambda: deque(maxlen=60))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=20))
    moving_frames: int = 0  # Сколько кадров объект реально двигался
    
    frames_seen: int = 0
    frames_lost: int = 0
    confirmed: bool = False
    
    # Флаг: показывать ли объект
    is_moving: bool = False
    
    def get_average_speed(self) -> float:
        if len(self.speed_history) < 2:
            return 0
        return float(np.mean(list(self.speed_history)))
    
    def update(self, centroid: Tuple[float, float], bbox: Tuple[int, int, int, int],
               mask: np.ndarray, min_speed: float = 2.0):
        prev_centroid = self.centroid
        
        self.centroid = centroid
        self.bbox = bbox
        self.mask = mask
        self.trajectory.append(centroid)
        
        # Вычисляем скорость
        if prev_centroid:
            speed = np.sqrt(
                (centroid[0] - prev_centroid[0])**2 +
                (centroid[1] - prev_centroid[1])**2
            )
            self.speed_history.append(speed)
            
            # Считаем кадры с движением
            if speed > min_speed:
                self.moving_frames += 1
            else:
                # Медленно уменьшаем счётчик
                self.moving_frames = max(0, self.moving_frames - 1)
        
        self.frames_seen += 1
        self.frames_lost = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 КОМБИНИРОВАННЫЙ ТРЕКЕР
# ═══════════════════════════════════════════════════════════════════════════════

class CombinedTracker:
    """
    Комбинированный трекер:
    - YOLO для людей
    - Сегментация для остального
    - Фильтр по скорости
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # YOLO
        self.yolo_model = None
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=40, detectShadows=True
        )
        
        # Объекты
        self.persons: Dict[int, TrackedPerson] = {}
        self.objects: Dict[int, TrackedObject] = {}
        self.next_person_id = 0
        self.next_object_id = 0
        
        # Цвета
        self.person_colors = self._generate_colors(20, saturation=0.9)
        self.object_colors = self._generate_colors(20, saturation=0.7, offset=0.5)
        
        # Статистика
        self.stats = {
            'raw_segments': 0,
            'moving_segments': 0,
            'persons': 0
        }
    
    def _generate_colors(self, n: int, saturation: float = 0.9, 
                        offset: float = 0.0) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(n):
            hue = ((i * 0.618033988749895) + offset) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, saturation, 1.0)
            colors.append(tuple(int(c * 255) for c in rgb[::-1]))
        return colors
    
    def _load_yolo(self):
        if self.yolo_model is not None:
            return
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(f'yolov8{self.config.yolo_model}-pose.pt')
            print(f"✓ YOLOv8{self.config.yolo_model}-pose loaded")
        except ImportError:
            print("⚠ ultralytics not installed")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[TrackedPerson], List[TrackedObject]]:
        """
        Обрабатывает кадр.
        Возвращает: (люди со скелетами, движущиеся объекты)
        """
        h, w = frame.shape[:2]
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 1: YOLO - детекция людей
        # ═══════════════════════════════════════════════════════════════════════
        
        self._load_yolo()
        
        yolo_results = self.yolo_model(frame, verbose=False, conf=self.config.yolo_confidence)
        
        person_detections = []
        person_masks = np.zeros((h, w), dtype=np.uint8)  # Маска людей
        
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
                    
                    # Добавляем в маску людей (чтобы исключить из сегментации)
                    cv2.rectangle(person_masks, (x1, y1), (x2, y2), 255, -1)
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
        # STAGE 2: Background Subtraction - сегментация
        # ═══════════════════════════════════════════════════════════════════════
        
        # Фильтр яркости
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_mask = np.ones_like(gray) * 255
        brightness_mask[gray < 15] = 0
        brightness_mask[gray > 245] = 0
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask[fg_mask == 127] = 0
        
        # Применяем фильтры
        fg_mask = cv2.bitwise_and(fg_mask, brightness_mask)
        
        # ИСКЛЮЧАЕМ области людей (уже обрабатываются YOLO)
        fg_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(person_masks))
        
        # Морфология
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Контуры
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
                'mask': obj_mask
            })
        
        self.stats['raw_segments'] = len(object_detections)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 3: Трекинг людей
        # ═══════════════════════════════════════════════════════════════════════
        
        for person in self.persons.values():
            person.frames_lost += 1
        
        self._match_persons(person_detections)
        
        for person in self.persons.values():
            if not person.confirmed and person.frames_seen >= self.config.confirm_frames:
                person.confirmed = True
        
        # Очистка
        to_remove = [pid for pid, p in self.persons.items() 
                    if p.frames_lost > self.config.lost_frames_max]
        for pid in to_remove:
            del self.persons[pid]
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 4: Трекинг объектов
        # ═══════════════════════════════════════════════════════════════════════
        
        for obj in self.objects.values():
            obj.frames_lost += 1
        
        self._match_objects(object_detections)
        
        for obj in self.objects.values():
            if not obj.confirmed and obj.frames_seen >= self.config.confirm_frames:
                obj.confirmed = True
        
        # Очистка
        to_remove = [oid for oid, o in self.objects.items()
                    if o.frames_lost > self.config.lost_frames_max]
        for oid in to_remove:
            del self.objects[oid]
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 5: Фильтр по скорости
        # ═══════════════════════════════════════════════════════════════════════
        
        for obj in self.objects.values():
            avg_speed = obj.get_average_speed()
            
            # Объект считается движущимся если:
            # 1. Средняя скорость выше порога
            # 2. ИЛИ он двигался достаточно кадров подряд
            obj.is_moving = (
                avg_speed > self.config.min_speed or
                obj.moving_frames >= self.config.speed_threshold_frames
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # Результаты
        # ═══════════════════════════════════════════════════════════════════════
        
        confirmed_persons = [p for p in self.persons.values() if p.confirmed]
        
        # Только ДВИЖУЩИЕСЯ объекты
        moving_objects = [o for o in self.objects.values() 
                         if o.confirmed and o.is_moving]
        
        self.stats['persons'] = len(confirmed_persons)
        self.stats['moving_segments'] = len(moving_objects)
        
        return confirmed_persons, moving_objects
    
    def _match_persons(self, detections: List[Dict]):
        """Сопоставляет YOLO детекции с людьми"""
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
        """Сопоставляет сегменты с объектами"""
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
                
                if dist < min_dist and dist < 100:
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

class CombinedVisualizer:
    """Визуализация скелетов и объектов"""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_skeleton(self, frame: np.ndarray, person: TrackedPerson,
                      min_conf: float = 0.3) -> np.ndarray:
        """Рисует скелет"""
        if person.keypoints is None:
            return frame
        
        kpts = person.keypoints
        confs = person.confidences if person.confidences is not None else np.ones(17)
        
        # Соединения
        for i, j in SKELETON_CONNECTIONS:
            if i >= len(kpts) or j >= len(kpts):
                continue
            
            pt1, pt2 = kpts[i], kpts[j]
            
            if pt1[0] <= 0 or pt2[0] <= 0:
                continue
            if confs[i] < min_conf or confs[j] < min_conf:
                continue
            
            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))
            
            color = get_limb_color(i, j)
            
            # Glow
            cv2.line(frame, pt1, pt2, tuple(c // 4 for c in color), 8)
            cv2.line(frame, pt1, pt2, tuple(c // 2 for c in color), 5)
            cv2.line(frame, pt1, pt2, color, 3)
            cv2.line(frame, pt1, pt2, (255, 255, 255), 1)
        
        # Точки
        for i, pt in enumerate(kpts):
            if pt[0] <= 0 or confs[i] < min_conf:
                continue
            
            pt = tuple(map(int, pt))
            color = person.color
            
            cv2.circle(frame, pt, 8, tuple(c // 3 for c in color), -1)
            cv2.circle(frame, pt, 5, color, -1)
            cv2.circle(frame, pt, 2, (255, 255, 255), -1)
        
        return frame
    
    def draw_person_info(self, frame: np.ndarray, person: TrackedPerson) -> np.ndarray:
        """Рисует метку и bbox человека"""
        if person.bbox is None:
            return frame
        
        x, y, w, h = person.bbox
        color = person.color
        
        # Уголки bbox
        corner = min(15, w // 5, h // 5)
        corners = [
            [(x, y), (x + corner, y)], [(x, y), (x, y + corner)],
            [(x + w, y), (x + w - corner, y)], [(x + w, y), (x + w, y + corner)],
            [(x, y + h), (x + corner, y + h)], [(x, y + h), (x, y + h - corner)],
            [(x + w, y + h), (x + w - corner, y + h)], [(x + w, y + h), (x + w, y + h - corner)],
        ]
        for p1, p2 in corners:
            cv2.line(frame, p1, p2, color, 2)
        
        # Метка
        speed = person.get_average_speed()
        label = f"Person #{person.id}"
        if speed > 1:
            label += f" v:{speed:.0f}"
        
        cv2.putText(frame, label, (x, y - 8), self.font, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, label, (x, y - 8), self.font, 0.5, color, 1)
        
        return frame
    
    def draw_object(self, frame: np.ndarray, obj: TrackedObject) -> np.ndarray:
        """Рисует движущийся объект"""
        if obj.mask is None:
            return frame
        
        color = obj.color
        
        # Маска
        overlay = frame.copy()
        overlay[obj.mask > 0] = color
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Контур
        contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 2)
        
        # Bbox
        if obj.bbox:
            x, y, w, h = obj.bbox
            corner = min(12, w // 5, h // 5)
            corners = [
                [(x, y), (x + corner, y)], [(x, y), (x, y + corner)],
                [(x + w, y), (x + w - corner, y)], [(x + w, y), (x + w, y + corner)],
                [(x, y + h), (x + corner, y + h)], [(x, y + h), (x, y + h - corner)],
                [(x + w, y + h), (x + w - corner, y + h)], [(x + w, y + h), (x + w, y + h - corner)],
            ]
            for p1, p2 in corners:
                cv2.line(frame, p1, p2, color, 2)
            
            # Метка
            speed = obj.get_average_speed()
            label = f"Object #{obj.id} v:{speed:.1f}"
            cv2.putText(frame, label, (x, y - 8), self.font, 0.45, (0, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 8), self.font, 0.45, color, 1)
        
        return frame
    
    def draw_trajectory(self, frame: np.ndarray, trajectory: deque,
                        color: Tuple[int, int, int]) -> np.ndarray:
        """Рисует траекторию"""
        if len(trajectory) < 2:
            return frame
        
        pts = list(trajectory)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            pt_color = tuple(int(c * alpha) for c in color)
            thickness = max(1, int(3 * alpha))
            
            pt1 = tuple(map(int, pts[i - 1]))
            pt2 = tuple(map(int, pts[i]))
            
            cv2.line(frame, pt1, pt2, pt_color, thickness)
        
        return frame
    
    def render(self, frame: np.ndarray, persons: List[TrackedPerson],
               objects: List[TrackedObject]) -> np.ndarray:
        """Полный рендер"""
        output = frame.copy()
        
        # Траектории
        for person in persons:
            output = self.draw_trajectory(output, person.trajectory, person.color)
        for obj in objects:
            output = self.draw_trajectory(output, obj.trajectory, obj.color)
        
        # Объекты (под скелетами)
        for obj in objects:
            output = self.draw_object(output, obj)
        
        # Скелеты
        for person in persons:
            output = self.draw_skeleton(output, person)
            output = self.draw_person_info(output, person)
        
        return output
    
    def draw_info_panel(self, frame: np.ndarray, tracker: CombinedTracker,
                        frame_idx: int, total_frames: int, fps: float) -> np.ndarray:
        """Рисует инфо-панель"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 35
        cv2.putText(frame, "COMBINED TRACKER", (20, y),
                   self.font, 0.65, (100, 200, 255), 2)
        
        y += 22
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames} | FPS: {fps:.1f}",
                   (20, y), self.font, 0.4, (200, 200, 200), 1)
        
        y += 20
        cv2.putText(frame, f"People (YOLO): {tracker.stats['persons']}",
                   (20, y), self.font, 0.4, (100, 255, 100), 1)
        
        y += 18
        cv2.putText(frame, f"Raw segments: {tracker.stats['raw_segments']}",
                   (20, y), self.font, 0.35, (150, 150, 150), 1)
        
        y += 18
        cv2.putText(frame, f"Moving objects: {tracker.stats['moving_segments']}",
                   (20, y), self.font, 0.4, (255, 200, 100), 1)
        
        y += 15
        progress = (frame_idx + 1) / total_frames
        bar_w = 280
        cv2.rectangle(frame, (20, y), (20 + bar_w, y + 6), (50, 50, 60), -1)
        cv2.rectangle(frame, (20, y), (20 + int(bar_w * progress), y + 6),
                     (100, 200, 255), -1)
        
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 ПРОЦЕССОР
# ═══════════════════════════════════════════════════════════════════════════════

class CombinedProcessor:
    """Главный процессор"""
    
    def __init__(self, config: Config):
        self.tracker = CombinedTracker(config)
        self.visualizer = CombinedVisualizer()
        self.config = config
    
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
        
        print(f"\n{'═'*60}")
        print(f"  🎭 COMBINED TRACKER")
        print(f"{'═'*60}")
        print(f"  📁 Input:  {input_path}")
        print(f"  📁 Output: {output_path}")
        print(f"  📐 Size:   {width}x{height} @ {fps:.1f} FPS")
        print(f"{'─'*60}")
        print(f"  🦴 YOLO:  yolov8{self.config.yolo_model}-pose")
        print(f"  🎯 Min speed for objects: {self.config.min_speed} px/frame")
        print(f"  ⏱  Speed window: {self.config.speed_window} frames")
        print(f"{'═'*60}")
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
            output_frame = self.visualizer.draw_info_panel(
                output_frame, self.tracker, frame_idx, total, current_fps
            )
            
            frame_time = time.time() - start
            frame_times.append(frame_time)
            
            out.write(output_frame)
            
            if show_preview:
                preview = cv2.resize(output_frame, (0, 0), fx=0.6, fy=0.6)
                cv2.imshow('Combined Tracker', preview)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  Interrupted")
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
            
            if frame_idx % 30 == 0:
                progress = (frame_idx + 1) / total * 100
                avg_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
                stats = self.tracker.stats
                print(f"\r  [{progress:5.1f}%] FPS: {avg_fps:.1f} | People: {stats['persons']} | Moving: {stats['moving_segments']}/{stats['raw_segments']}", end="")
            
            frame_idx += 1
        
        print(f"\n\n{'═'*60}")
        print(f"  ✅ COMPLETE")
        print(f"{'═'*60}")
        if frame_times:
            print(f"  ⏱  Avg time: {np.mean(frame_times)*1000:.1f} ms")
            print(f"  🚀 Avg FPS: {1.0/np.mean(frame_times):.1f}")
        print(f"  💾 Saved: {output_path}")
        print(f"{'═'*60}\n")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🎭 Combined Tracker: YOLO Skeletons + Moving Objects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python combined_tracker.py test.mp4
  python combined_tracker.py test.mp4 --min-speed 5
  python combined_tracker.py test.mp4 --model m --min-speed 3
        """
    )
    
    parser.add_argument('input', help='Input video')
    parser.add_argument('-o', '--output', default=None)
    
    # YOLO
    parser.add_argument('--model', '-m', default='n',
                       choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--conf', type=float, default=0.3)
    
    # Speed filter (ГЛАВНОЕ)
    parser.add_argument('--min-speed', type=float, default=2.0,
                       help='Minimum speed to show object (px/frame)')
    parser.add_argument('--speed-window', type=int, default=10,
                       help='Frames to average speed')
    parser.add_argument('--speed-frames', type=int, default=5,
                       help='Frames object must be moving')
    
    parser.add_argument('--min-area', type=int, default=1500)
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--max-frames', type=int, default=None)
    
    args = parser.parse_args()
    
    config = Config(
        yolo_model=args.model,
        yolo_confidence=args.conf,
        min_area=args.min_area,
        min_speed=args.min_speed,
        speed_window=args.speed_window,
        speed_threshold_frames=args.speed_frames
    )
    
    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_combined.mp4")
    
    processor = CombinedProcessor(config)
    processor.process_video(
        args.input,
        args.output,
        show_preview=not args.no_preview,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()


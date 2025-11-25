#!/usr/bin/env python3
"""
🎬 Demo: Object Tracker with Beautiful Visualization
=====================================================
Быстрый демо-скрипт для тестирования трекинга.
Работает без установки дополнительных ML библиотек.

Запуск:
    python demo_tracker.py test.mp4
    python demo_tracker.py test.mp4 --pose  # С детекцией скелетов
"""

import cv2
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import colorsys
import time
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 СТИЛЬНАЯ ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

class NeonStyle:
    """Неоновый визуальный стиль"""
    
    # Цветовая схема
    BACKGROUND = (15, 15, 20)
    ACCENT_CYAN = (255, 220, 80)  # BGR: яркий cyan
    ACCENT_MAGENTA = (255, 80, 220)  # BGR: magenta
    ACCENT_GREEN = (80, 255, 180)  # BGR: mint green
    ACCENT_ORANGE = (80, 180, 255)  # BGR: orange
    ACCENT_PURPLE = (255, 100, 200)  # BGR: purple
    
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (180, 180, 190)
    
    @staticmethod
    def generate_neon_palette(n: int) -> List[Tuple[int, int, int]]:
        """Генерирует неоновую палитру"""
        base_hues = [0.5, 0.85, 0.35, 0.1, 0.65, 0.95, 0.45, 0.75]
        colors = []
        
        for i in range(n):
            hue = base_hues[i % len(base_hues)]
            hue += (i // len(base_hues)) * 0.05  # Небольшой сдвиг для вариации
            hue = hue % 1.0
            
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
            # BGR для OpenCV
            colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
        
        return colors


class FuturisticRenderer:
    """Футуристический рендерер с неоновыми эффектами"""
    
    def __init__(self):
        self.colors = NeonStyle.generate_neon_palette(20)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_neon_line(self, img: np.ndarray, pt1: Tuple[int, int], 
                       pt2: Tuple[int, int], color: Tuple[int, int, int],
                       thickness: int = 2, glow: bool = True):
        """Рисует линию с неоновым свечением"""
        if glow:
            # Внешнее свечение
            for i, alpha in enumerate([0.1, 0.2, 0.4]):
                glow_thickness = thickness + 8 - i * 2
                glow_color = tuple(int(c * alpha) for c in color)
                cv2.line(img, pt1, pt2, glow_color, glow_thickness)
        
        # Основная линия
        cv2.line(img, pt1, pt2, color, thickness)
        
        # Яркий центр
        if thickness > 1:
            bright_color = tuple(min(255, int(c * 1.2)) for c in color)
            cv2.line(img, pt1, pt2, bright_color, max(1, thickness - 1))
    
    def draw_neon_circle(self, img: np.ndarray, center: Tuple[int, int],
                         radius: int, color: Tuple[int, int, int],
                         thickness: int = -1, glow: bool = True):
        """Рисует круг с неоновым свечением"""
        if glow:
            for i, (alpha, extra_r) in enumerate([(0.1, 8), (0.2, 5), (0.3, 2)]):
                glow_color = tuple(int(c * alpha) for c in color)
                cv2.circle(img, center, radius + extra_r, glow_color, -1)
        
        cv2.circle(img, center, radius, color, thickness)
        
        # Блик
        if thickness == -1 and radius > 3:
            highlight_pos = (center[0] - radius // 3, center[1] - radius // 3)
            cv2.circle(img, highlight_pos, max(1, radius // 4), (255, 255, 255), -1)
    
    def draw_futuristic_box(self, img: np.ndarray, bbox: Tuple[int, int, int, int],
                            color: Tuple[int, int, int], label: str = ""):
        """Рисует футуристический bounding box"""
        x, y, w, h = bbox
        
        # Длина уголков
        corner_len = min(25, w // 4, h // 4)
        thickness = 2
        
        # Полупрозрачный фон
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)
        
        # Уголки с glow
        corners = [
            # Верхний левый
            [(x, y), (x + corner_len, y)],
            [(x, y), (x, y + corner_len)],
            # Верхний правый
            [(x + w, y), (x + w - corner_len, y)],
            [(x + w, y), (x + w, y + corner_len)],
            # Нижний левый
            [(x, y + h), (x + corner_len, y + h)],
            [(x, y + h), (x, y + h - corner_len)],
            # Нижний правый
            [(x + w, y + h), (x + w - corner_len, y + h)],
            [(x + w, y + h), (x + w, y + h - corner_len)],
        ]
        
        for pt1, pt2 in corners:
            self.draw_neon_line(img, pt1, pt2, color, thickness)
        
        # Метка
        if label:
            # Фон для текста
            (text_w, text_h), _ = cv2.getTextSize(label, self.font, 0.5, 1)
            padding = 5
            
            # Подложка
            cv2.rectangle(img, 
                         (x, y - text_h - 2 * padding),
                         (x + text_w + 2 * padding, y),
                         color, -1)
            
            # Текст
            cv2.putText(img, label, (x + padding, y - padding),
                       self.font, 0.5, (0, 0, 0), 2)
            cv2.putText(img, label, (x + padding, y - padding),
                       self.font, 0.5, (255, 255, 255), 1)
    
    def draw_trajectory(self, img: np.ndarray, points: List[Tuple[int, int]],
                        color: Tuple[int, int, int], fade: bool = True):
        """Рисует траекторию с эффектом затухания"""
        if len(points) < 2:
            return
        
        for i in range(1, len(points)):
            if fade:
                alpha = (i / len(points)) ** 0.7  # Нелинейное затухание
                pt_color = tuple(int(c * alpha) for c in color)
                thickness = max(1, int(3 * alpha))
            else:
                pt_color = color
                thickness = 2
            
            pt1 = tuple(map(int, points[i - 1]))
            pt2 = tuple(map(int, points[i]))
            
            self.draw_neon_line(img, pt1, pt2, pt_color, thickness, glow=fade)


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 ИНФОРМАЦИОННАЯ ПАНЕЛЬ
# ═══════════════════════════════════════════════════════════════════════════════

class InfoPanel:
    """Стильная информационная панель"""
    
    def __init__(self, width: int = 280, height: int = 720):
        self.width = width
        self.height = height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.renderer = FuturisticRenderer()
        
    def render(self, objects: List['TrackedObject'], frame_idx: int,
               total_frames: int, fps: float) -> np.ndarray:
        """Рендерит панель"""
        panel = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Градиентный фон
        for i in range(self.width):
            alpha = 1.0 - (i / self.width) * 0.5
            panel[:, i] = (int(20 * alpha), int(18 * alpha), int(25 * alpha))
        
        y = 35
        
        # Заголовок
        cv2.putText(panel, "OBJECT TRACKER", (15, y),
                   self.font, 0.7, NeonStyle.ACCENT_CYAN, 2)
        y += 10
        
        # Декоративная линия
        cv2.line(panel, (15, y), (self.width - 15, y), NeonStyle.ACCENT_CYAN, 1)
        y += 25
        
        # Статистика
        stats = [
            ("FRAME", f"{frame_idx + 1}/{total_frames}"),
            ("FPS", f"{fps:.1f}"),
            ("OBJECTS", f"{len([o for o in objects if o.active])}"),
            ("PROGRESS", f"{(frame_idx + 1) / total_frames * 100:.0f}%"),
        ]
        
        for label, value in stats:
            cv2.putText(panel, label, (20, y),
                       self.font, 0.4, NeonStyle.TEXT_SECONDARY, 1)
            cv2.putText(panel, value, (110, y),
                       self.font, 0.45, NeonStyle.TEXT_PRIMARY, 1)
            y += 22
        
        y += 15
        
        # Progress bar
        progress = (frame_idx + 1) / total_frames
        bar_width = self.width - 40
        cv2.rectangle(panel, (20, y), (20 + bar_width, y + 8),
                     (40, 40, 50), -1)
        cv2.rectangle(panel, (20, y), (20 + int(bar_width * progress), y + 8),
                     NeonStyle.ACCENT_CYAN, -1)
        y += 30
        
        # Список объектов
        cv2.putText(panel, "TRACKED OBJECTS", (15, y),
                   self.font, 0.5, NeonStyle.ACCENT_GREEN, 1)
        y += 8
        cv2.line(panel, (15, y), (self.width - 15, y), (50, 50, 60), 1)
        y += 20
        
        for obj in objects[:8]:  # Максимум 8 объектов
            if not obj.active:
                continue
            
            # Цветной маркер
            self.renderer.draw_neon_circle(panel, (30, y - 4), 6, obj.color, -1, glow=False)
            
            # Имя
            name = f"Object {obj.id}"
            cv2.putText(panel, name, (50, y),
                       self.font, 0.4, NeonStyle.TEXT_PRIMARY, 1)
            
            # Скорость
            vel = obj.get_velocity()
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            cv2.putText(panel, f"v: {speed:.1f}", (50, y + 15),
                       self.font, 0.35, NeonStyle.TEXT_SECONDARY, 1)
            
            # Направление
            if speed > 1:
                angle = np.arctan2(vel[1], vel[0]) * 180 / np.pi
                direction = self._get_direction(angle)
                cv2.putText(panel, direction, (130, y + 15),
                           self.font, 0.35, obj.color, 1)
            
            y += 45
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(panel, timestamp, (self.width - 80, self.height - 20),
                   self.font, 0.4, NeonStyle.TEXT_SECONDARY, 1)
        
        return panel
    
    def _get_direction(self, angle: float) -> str:
        """Возвращает текстовое направление"""
        if -22.5 <= angle < 22.5:
            return "→"
        elif 22.5 <= angle < 67.5:
            return "↘"
        elif 67.5 <= angle < 112.5:
            return "↓"
        elif 112.5 <= angle < 157.5:
            return "↙"
        elif angle >= 157.5 or angle < -157.5:
            return "←"
        elif -157.5 <= angle < -112.5:
            return "↖"
        elif -112.5 <= angle < -67.5:
            return "↑"
        else:
            return "↗"


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 ТРЕКЕР ОБЪЕКТОВ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedObject:
    """Отслеживаемый объект"""
    id: int
    color: Tuple[int, int, int]
    trajectory: deque = field(default_factory=lambda: deque(maxlen=80))
    bbox: Optional[Tuple[int, int, int, int]] = None
    centroid: Optional[Tuple[int, int]] = None
    mask: Optional[np.ndarray] = None
    active: bool = True
    lost_frames: int = 0
    _prev_centroid: Optional[Tuple[int, int]] = None
    
    def update(self, bbox: Tuple[int, int, int, int], 
               centroid: Tuple[int, int],
               mask: Optional[np.ndarray] = None):
        """Обновляет состояние объекта"""
        self._prev_centroid = self.centroid
        self.bbox = bbox
        self.centroid = centroid
        self.mask = mask
        self.trajectory.append(centroid)
        self.lost_frames = 0
        self.active = True
    
    def get_velocity(self) -> Tuple[float, float]:
        """Возвращает текущую скорость"""
        if len(self.trajectory) < 2:
            return (0, 0)
        
        recent = list(self.trajectory)[-5:]
        if len(recent) < 2:
            return (0, 0)
        
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        return (dx / len(recent), dy / len(recent))


class MotionTracker:
    """Трекер движущихся объектов на основе детекции движения"""
    
    def __init__(self, min_area: int = 800, max_objects: int = 15):
        self.min_area = min_area
        self.max_objects = max_objects
        
        # Background subtractor с улучшенными параметрами
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=400,
            varThreshold=40,
            detectShadows=True
        )
        
        self.objects: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.colors = NeonStyle.generate_neon_palette(30)
        
        # Параметры сопоставления
        self.max_distance = 120
        self.max_lost_frames = 15
        
    def process_frame(self, frame: np.ndarray) -> List[TrackedObject]:
        """Обрабатывает кадр и возвращает список объектов"""
        # Применяем background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Удаляем тени (значение 127)
        fg_mask[fg_mask == 127] = 0
        
        # Морфологические операции
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Находим контуры
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Извлекаем детекции
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Фильтруем слишком узкие/широкие объекты
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5:
                continue
            
            # Центроид
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # Создаем маску для этого объекта
            obj_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(obj_mask, [contour], -1, 255, -1)
            
            detections.append({
                'bbox': (x, y, w, h),
                'centroid': (cx, cy),
                'mask': obj_mask,
                'area': area
            })
        
        # Сортируем по площади (большие первые)
        detections.sort(key=lambda d: d['area'], reverse=True)
        detections = detections[:self.max_objects]
        
        # Сопоставляем с существующими объектами
        self._match_detections(detections)
        
        # Удаляем потерянные объекты
        self._cleanup_lost_objects()
        
        return list(self.objects.values())
    
    def _match_detections(self, detections: List[Dict]):
        """Сопоставляет детекции с объектами"""
        # Увеличиваем счетчик потерянных кадров для всех объектов
        for obj in self.objects.values():
            obj.lost_frames += 1
        
        if not detections:
            return
        
        # Получаем активные объекты
        active_objs = [o for o in self.objects.values() 
                      if o.centroid is not None and o.lost_frames < self.max_lost_frames]
        
        used_detections = set()
        
        # Сопоставляем по расстоянию
        for obj in active_objs:
            min_dist = float('inf')
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                
                dist = np.sqrt(
                    (obj.centroid[0] - det['centroid'][0])**2 +
                    (obj.centroid[1] - det['centroid'][1])**2
                )
                
                # Учитываем предсказанное положение
                vel = obj.get_velocity()
                predicted = (obj.centroid[0] + vel[0], obj.centroid[1] + vel[1])
                pred_dist = np.sqrt(
                    (predicted[0] - det['centroid'][0])**2 +
                    (predicted[1] - det['centroid'][1])**2
                )
                
                combined_dist = min(dist, pred_dist)
                
                if combined_dist < min_dist and combined_dist < self.max_distance:
                    min_dist = combined_dist
                    best_det_idx = i
            
            if best_det_idx >= 0:
                det = detections[best_det_idx]
                obj.update(det['bbox'], det['centroid'], det['mask'])
                used_detections.add(best_det_idx)
        
        # Создаем новые объекты для несопоставленных детекций
        for i, det in enumerate(detections):
            if i not in used_detections:
                if len(self.objects) < self.max_objects:
                    self._create_object(det)
    
    def _create_object(self, detection: Dict) -> TrackedObject:
        """Создает новый объект"""
        obj = TrackedObject(
            id=self.next_id,
            color=self.colors[self.next_id % len(self.colors)]
        )
        obj.update(detection['bbox'], detection['centroid'], detection['mask'])
        self.objects[self.next_id] = obj
        self.next_id += 1
        return obj
    
    def _cleanup_lost_objects(self):
        """Удаляет потерянные объекты"""
        to_remove = []
        for obj_id, obj in self.objects.items():
            if obj.lost_frames >= self.max_lost_frames:
                obj.active = False
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.objects[obj_id]


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 ГЛАВНЫЙ ПРОЦЕССОР
# ═══════════════════════════════════════════════════════════════════════════════

class DemoProcessor:
    """Главный процессор для демо"""
    
    def __init__(self, show_panel: bool = True, use_pose: bool = False):
        self.show_panel = show_panel
        self.use_pose = use_pose
        
        self.tracker = MotionTracker()
        self.renderer = FuturisticRenderer()
        self.panel = InfoPanel()
        
        # YOLO pose (если включено)
        self.pose_model = None
        if use_pose:
            self._init_pose()
    
    def _init_pose(self):
        """Инициализирует YOLO-Pose"""
        try:
            from ultralytics import YOLO
            self.pose_model = YOLO('yolov8n-pose.pt')
            print("✓ YOLOv8-Pose загружена")
        except ImportError:
            print("⚠ ultralytics не установлена, pose detection отключён")
            self.use_pose = False
    
    def process_video(self, input_path: str, output_path: Optional[str] = None,
                      show_preview: bool = True, max_frames: Optional[int] = None):
        """Обрабатывает видео"""
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Параметры видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        # Настраиваем панель
        self.panel.height = height
        
        # Выходное видео
        output_width = width + (self.panel.width if self.show_panel else 0)
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))
        
        print(f"\n{'═'*60}")
        print(f"  🎬 OBJECT TRACKING DEMO")
        print(f"{'═'*60}")
        print(f"  📁 Input:  {input_path}")
        print(f"  📐 Size:   {width}x{height}")
        print(f"  🎞  FPS:    {fps:.1f}")
        print(f"  📊 Frames: {total_frames}")
        if output_path:
            print(f"  💾 Output: {output_path}")
        print(f"{'═'*60}")
        print(f"\n  Press 'Q' to quit, 'P' to pause\n")
        
        frame_times = []
        frame_idx = 0
        paused = False
        
        while frame_idx < total_frames:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Трекинг
                objects = self.tracker.process_frame(frame)
                
                # Рендеринг
                output_frame = self._render_frame(frame, objects, frame_idx, 
                                                  total_frames, fps, frame_times)
                
                # Pose detection
                if self.use_pose and self.pose_model:
                    output_frame = self._render_poses(output_frame[:, :width], 
                                                     output_frame[:, width:] if self.show_panel else None)
                
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                
                # Сохраняем
                if out:
                    out.write(output_frame)
                
                frame_idx += 1
                
                # Прогресс
                if frame_idx % 30 == 0:
                    progress = frame_idx / total_frames * 100
                    avg_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
                    print(f"\r  Processing: {progress:.1f}% | FPS: {avg_fps:.1f}", end="")
            
            # Превью
            if show_preview:
                preview = cv2.resize(output_frame, (0, 0), fx=0.6, fy=0.6)
                cv2.imshow('Object Tracker Demo', preview)
                
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                if key == ord('q'):
                    print("\n\n  ⚠ Interrupted by user")
                    break
                elif key == ord('p') or key == ord(' '):
                    paused = not paused
        
        print(f"\n\n{'═'*60}")
        print(f"  ✅ PROCESSING COMPLETE")
        print(f"{'═'*60}")
        
        if frame_times:
            avg_time = np.mean(frame_times)
            print(f"  ⏱  Avg frame time: {avg_time*1000:.1f} ms")
            print(f"  🚀 Processing FPS: {1.0/avg_time:.1f}")
        
        if output_path:
            print(f"  💾 Saved to: {output_path}")
        
        print(f"{'═'*60}\n")
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
    
    def _render_frame(self, frame: np.ndarray, objects: List[TrackedObject],
                      frame_idx: int, total_frames: int, fps: float,
                      frame_times: List[float]) -> np.ndarray:
        """Рендерит кадр с визуализацией"""
        output = frame.copy()
        
        # Рендерим маски
        for obj in objects:
            if not obj.active or obj.mask is None:
                continue
            
            # Полупрозрачная маска
            mask_overlay = output.copy()
            mask_overlay[obj.mask > 0] = obj.color
            cv2.addWeighted(mask_overlay, 0.3, output, 0.7, 0, output)
            
            # Контур
            contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, obj.color, 2)
        
        # Рендерим траектории
        for obj in objects:
            if not obj.active or len(obj.trajectory) < 2:
                continue
            self.renderer.draw_trajectory(output, list(obj.trajectory), obj.color)
        
        # Рендерим bboxes
        for obj in objects:
            if not obj.active or obj.bbox is None:
                continue
            
            label = f"ID:{obj.id}"
            vel = obj.get_velocity()
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            if speed > 1:
                label += f" v:{speed:.0f}"
            
            self.renderer.draw_futuristic_box(output, obj.bbox, obj.color, label)
        
        # Рендерим центроиды
        for obj in objects:
            if not obj.active or obj.centroid is None:
                continue
            self.renderer.draw_neon_circle(output, obj.centroid, 5, obj.color)
        
        # Панель
        if self.show_panel:
            current_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else fps
            panel = self.panel.render(objects, frame_idx, total_frames, current_fps)
            output = np.hstack([output, panel])
        
        return output
    
    def _render_poses(self, frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
        """Рендерит позы (скелеты)"""
        if self.pose_model is None:
            return np.hstack([frame, panel]) if panel is not None else frame
        
        results = self.pose_model(frame, verbose=False)
        
        # Соединения скелета
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Голова
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Руки
            (5, 11), (6, 12), (11, 12),  # Торс
            (11, 13), (13, 15), (12, 14), (14, 16)  # Ноги
        ]
        
        for result in results:
            if result.keypoints is None:
                continue
            
            keypoints = result.keypoints.xy.cpu().numpy()
            confs = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
            
            for person_idx, kpts in enumerate(keypoints):
                color = self.renderer.colors[person_idx % len(self.renderer.colors)]
                
                # Рисуем соединения
                for i, j in skeleton:
                    if i < len(kpts) and j < len(kpts):
                        pt1 = tuple(map(int, kpts[i]))
                        pt2 = tuple(map(int, kpts[j]))
                        
                        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                            if confs is not None:
                                if confs[person_idx][i] < 0.5 or confs[person_idx][j] < 0.5:
                                    continue
                            
                            self.renderer.draw_neon_line(frame, pt1, pt2, color, 2)
                
                # Рисуем точки
                for i, pt in enumerate(kpts):
                    pt = tuple(map(int, pt))
                    if pt[0] > 0 and pt[1] > 0:
                        if confs is not None and confs[person_idx][i] < 0.5:
                            continue
                        self.renderer.draw_neon_circle(frame, pt, 4, color, -1, glow=False)
        
        if panel is not None:
            return np.hstack([frame, panel])
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🎬 Demo: Object Tracker with Beautiful Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_tracker.py test.mp4
  python demo_tracker.py test.mp4 -o output.mp4
  python demo_tracker.py test.mp4 --pose
  python demo_tracker.py test.mp4 --no-panel --max-frames 500
        """
    )
    
    parser.add_argument('input', type=str, help='Input video path')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output video path (default: input_tracked.mp4)')
    parser.add_argument('--pose', action='store_true',
                       help='Enable pose estimation (requires ultralytics)')
    parser.add_argument('--no-panel', action='store_true',
                       help='Disable info panel')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable preview window')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    
    args = parser.parse_args()
    
    # Выходной путь
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_tracked.mp4")
    
    # Создаем процессор
    processor = DemoProcessor(
        show_panel=not args.no_panel,
        use_pose=args.pose
    )
    
    # Обрабатываем видео
    processor.process_video(
        input_path=args.input,
        output_path=args.output,
        show_preview=not args.no_preview,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()


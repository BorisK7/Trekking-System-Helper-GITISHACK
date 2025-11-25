#!/usr/bin/env python3
"""
🦴 YOLO Skeleton Tracker
========================
Простой и надёжный трекер людей с отрисовкой скелетов.
Использует YOLOv8-pose.

Запуск:
    python yolo_skeleton_tracker.py test.mp4
    python yolo_skeleton_tracker.py test.mp4 --model m  # medium модель
    python yolo_skeleton_tracker.py test.mp4 --conf 0.5
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

# Соединения скелета COCO
SKELETON_CONNECTIONS = [
    # Голова
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Торс
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Левая рука
    (5, 7), (7, 9),
    # Правая рука
    (6, 8), (8, 10),
    # Левая нога
    (11, 13), (13, 15),
    # Правая нога
    (12, 14), (14, 16)
]

# Названия точек
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Цвета для частей тела
LIMB_COLORS = {
    'head': (255, 200, 100),      # Жёлтый
    'torso': (100, 255, 100),     # Зелёный
    'left_arm': (255, 100, 100),  # Красный
    'right_arm': (100, 100, 255), # Синий
    'left_leg': (255, 100, 255),  # Розовый
    'right_leg': (100, 255, 255), # Голубой
}


def get_limb_color(connection: Tuple[int, int]) -> Tuple[int, int, int]:
    """Возвращает цвет для соединения"""
    i, j = connection
    
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
    else:
        return LIMB_COLORS['right_leg']


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 ТРЕКЕР
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedPerson:
    """Отслеживаемый человек"""
    id: int
    color: Tuple[int, int, int]
    
    # Текущее состояние
    keypoints: Optional[np.ndarray] = None
    confidences: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    
    # История
    trajectory: deque = field(default_factory=lambda: deque(maxlen=60))
    
    # Temporal
    frames_seen: int = 0
    frames_lost: int = 0
    confirmed: bool = False
    
    def get_center(self) -> Optional[Tuple[float, float]]:
        """Возвращает центр человека (по бёдрам или bbox)"""
        if self.keypoints is not None:
            # Центр между бёдрами (точки 11 и 12)
            left_hip = self.keypoints[11]
            right_hip = self.keypoints[12]
            
            if left_hip[0] > 0 and right_hip[0] > 0:
                return ((left_hip[0] + right_hip[0]) / 2,
                       (left_hip[1] + right_hip[1]) / 2)
            
            # Fallback: центр плеч
            left_shoulder = self.keypoints[5]
            right_shoulder = self.keypoints[6]
            
            if left_shoulder[0] > 0 and right_shoulder[0] > 0:
                return ((left_shoulder[0] + right_shoulder[0]) / 2,
                       (left_shoulder[1] + right_shoulder[1]) / 2)
        
        if self.bbox:
            x, y, w, h = self.bbox
            return (x + w / 2, y + h / 2)
        
        return None
    
    def update(self, keypoints: np.ndarray, confidences: np.ndarray,
               bbox: Tuple[int, int, int, int]):
        """Обновляет состояние"""
        self.keypoints = keypoints
        self.confidences = confidences
        self.bbox = bbox
        
        center = self.get_center()
        if center:
            self.trajectory.append(center)
        
        self.frames_seen += 1
        self.frames_lost = 0


class YOLOPoseTracker:
    """Трекер людей с YOLO-pose"""
    
    def __init__(self, model_size: str = 'n', confidence: float = 0.3):
        """
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x'
            confidence: Минимальная confidence
        """
        self.model_size = model_size
        self.confidence = confidence
        self.model = None
        
        self.persons: Dict[int, TrackedPerson] = {}
        self.next_id = 0
        self.colors = self._generate_colors(30)
        
        self.confirm_frames = 2
        self.lost_frames_max = 15
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Генерирует яркие цвета"""
        colors = []
        for i in range(n):
            hue = (i * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
            colors.append(tuple(int(c * 255) for c in rgb[::-1]))
        return colors
    
    def _load_model(self):
        """Загружает YOLO модель"""
        if self.model is not None:
            return
        
        try:
            from ultralytics import YOLO
            model_name = f'yolov8{self.model_size}-pose.pt'
            self.model = YOLO(model_name)
            print(f"✓ Загружена модель {model_name}")
        except ImportError:
            print("❌ ultralytics не установлена!")
            print("   Установите: pip install ultralytics")
            raise
    
    def process_frame(self, frame: np.ndarray) -> List[TrackedPerson]:
        """Обрабатывает кадр"""
        self._load_model()
        
        # Детекция
        results = self.model(frame, verbose=False, conf=self.confidence)
        
        # Извлекаем людей
        detections = []
        for result in results:
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
                else:
                    # Вычисляем bbox по точкам
                    valid_pts = kpts[kpts[:, 0] > 0]
                    if len(valid_pts) > 0:
                        x1, y1 = valid_pts.min(axis=0)
                        x2, y2 = valid_pts.max(axis=0)
                        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    else:
                        continue
                
                detections.append({
                    'keypoints': kpts,
                    'confidences': conf,
                    'bbox': bbox
                })
        
        # Трекинг
        for person in self.persons.values():
            person.frames_lost += 1
        
        self._match_detections(detections)
        
        # Подтверждение
        for person in self.persons.values():
            if not person.confirmed and person.frames_seen >= self.confirm_frames:
                person.confirmed = True
        
        # Очистка
        self._cleanup()
        
        return [p for p in self.persons.values() if p.confirmed]
    
    def _match_detections(self, detections: List[Dict]):
        """Сопоставляет детекции с людьми"""
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
                
                # Центр детекции
                x, y, w, h = det['bbox']
                det_center = (x + w / 2, y + h / 2)
                
                dist = np.sqrt(
                    (center[0] - det_center[0])**2 +
                    (center[1] - det_center[1])**2
                )
                
                if dist < min_dist and dist < 150:
                    min_dist = dist
                    best_idx = i
            
            if best_idx >= 0:
                det = detections[best_idx]
                person.update(det['keypoints'], det['confidences'], det['bbox'])
                used.add(best_idx)
        
        # Новые люди
        for i, det in enumerate(detections):
            if i not in used:
                self._create_person(det)
    
    def _create_person(self, detection: Dict):
        """Создаёт нового человека"""
        person = TrackedPerson(
            id=self.next_id,
            color=self.colors[self.next_id % len(self.colors)]
        )
        person.update(
            detection['keypoints'],
            detection['confidences'],
            detection['bbox']
        )
        self.persons[self.next_id] = person
        self.next_id += 1
    
    def _cleanup(self):
        """Удаляет потерянных людей"""
        to_remove = [
            pid for pid, p in self.persons.items()
            if p.frames_lost > self.lost_frames_max
        ]
        for pid in to_remove:
            del self.persons[pid]


# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

class SkeletonVisualizer:
    """Красивая отрисовка скелетов"""
    
    def __init__(self, style: str = 'neon'):
        """
        Args:
            style: 'neon', 'classic', 'minimal'
        """
        self.style = style
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_skeleton(self, frame: np.ndarray, person: TrackedPerson,
                      min_conf: float = 0.3) -> np.ndarray:
        """Рисует скелет человека"""
        if person.keypoints is None:
            return frame
        
        kpts = person.keypoints
        confs = person.confidences if person.confidences is not None else np.ones(17)
        color = person.color
        
        # Рисуем соединения
        for i, j in SKELETON_CONNECTIONS:
            if i >= len(kpts) or j >= len(kpts):
                continue
            
            pt1 = kpts[i]
            pt2 = kpts[j]
            
            # Проверяем валидность точек
            if pt1[0] <= 0 or pt1[1] <= 0 or pt2[0] <= 0 or pt2[1] <= 0:
                continue
            
            # Проверяем confidence
            if confs[i] < min_conf or confs[j] < min_conf:
                continue
            
            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))
            
            if self.style == 'neon':
                # Glow эффект
                limb_color = get_limb_color((i, j))
                
                # Внешнее свечение
                cv2.line(frame, pt1, pt2, tuple(c // 4 for c in limb_color), 8)
                cv2.line(frame, pt1, pt2, tuple(c // 2 for c in limb_color), 5)
                # Основная линия
                cv2.line(frame, pt1, pt2, limb_color, 3)
                # Яркий центр
                cv2.line(frame, pt1, pt2, (255, 255, 255), 1)
                
            elif self.style == 'classic':
                cv2.line(frame, pt1, pt2, color, 3)
                
            else:  # minimal
                cv2.line(frame, pt1, pt2, color, 2)
        
        # Рисуем точки
        for i, pt in enumerate(kpts):
            if pt[0] <= 0 or pt[1] <= 0:
                continue
            if confs[i] < min_conf:
                continue
            
            pt = tuple(map(int, pt))
            
            if self.style == 'neon':
                # Glow
                cv2.circle(frame, pt, 8, tuple(c // 3 for c in color), -1)
                cv2.circle(frame, pt, 5, color, -1)
                cv2.circle(frame, pt, 2, (255, 255, 255), -1)
            else:
                cv2.circle(frame, pt, 4, (255, 255, 255), -1)
                cv2.circle(frame, pt, 3, color, -1)
        
        return frame
    
    def draw_trajectory(self, frame: np.ndarray, person: TrackedPerson) -> np.ndarray:
        """Рисует траекторию движения"""
        if len(person.trajectory) < 2:
            return frame
        
        pts = list(person.trajectory)
        color = person.color
        
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            pt_color = tuple(int(c * alpha) for c in color)
            thickness = max(1, int(3 * alpha))
            
            pt1 = tuple(map(int, pts[i - 1]))
            pt2 = tuple(map(int, pts[i]))
            
            if self.style == 'neon':
                # Glow
                cv2.line(frame, pt1, pt2, tuple(c // 3 for c in pt_color), thickness + 4)
            
            cv2.line(frame, pt1, pt2, pt_color, thickness)
        
        return frame
    
    def draw_bbox(self, frame: np.ndarray, person: TrackedPerson) -> np.ndarray:
        """Рисует bounding box"""
        if person.bbox is None:
            return frame
        
        x, y, w, h = person.bbox
        color = person.color
        
        # Уголки
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
            if self.style == 'neon':
                cv2.line(frame, p1, p2, tuple(c // 2 for c in color), 4)
            cv2.line(frame, p1, p2, color, 2)
        
        return frame
    
    def draw_label(self, frame: np.ndarray, person: TrackedPerson) -> np.ndarray:
        """Рисует метку"""
        if person.bbox is None:
            return frame
        
        x, y, w, h = person.bbox
        color = person.color
        
        label = f"Person #{person.id}"
        
        (text_w, text_h), _ = cv2.getTextSize(label, self.font, 0.6, 2)
        
        # Фон
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
        
        # Текст
        cv2.putText(frame, label, (x + 5, y - 5),
                   self.font, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, label, (x + 5, y - 5),
                   self.font, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def render(self, frame: np.ndarray, persons: List[TrackedPerson],
               show_skeleton: bool = True,
               show_trajectory: bool = True,
               show_bbox: bool = True,
               show_label: bool = True) -> np.ndarray:
        """Полный рендер"""
        output = frame.copy()
        
        # Сначала траектории (под скелетами)
        if show_trajectory:
            for person in persons:
                output = self.draw_trajectory(output, person)
        
        # Скелеты
        if show_skeleton:
            for person in persons:
                output = self.draw_skeleton(output, person)
        
        # Bbox
        if show_bbox:
            for person in persons:
                output = self.draw_bbox(output, person)
        
        # Метки
        if show_label:
            for person in persons:
                output = self.draw_label(output, person)
        
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 ИНФО-ПАНЕЛЬ
# ═══════════════════════════════════════════════════════════════════════════════

class InfoPanel:
    """Информационная панель"""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw(self, frame: np.ndarray, persons: List[TrackedPerson],
             frame_idx: int, total_frames: int, fps: float) -> np.ndarray:
        
        h, w = frame.shape[:2]
        
        # Фон
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 35
        cv2.putText(frame, "YOLO SKELETON TRACKER", (20, y),
                   self.font, 0.6, (100, 200, 255), 2)
        
        y += 25
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames}", (20, y),
                   self.font, 0.45, (200, 200, 200), 1)
        
        y += 20
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y),
                   self.font, 0.45, (200, 200, 200), 1)
        
        y += 20
        cv2.putText(frame, f"People: {len(persons)}", (20, y),
                   self.font, 0.45, (100, 255, 100), 1)
        
        # Progress bar
        y += 15
        progress = (frame_idx + 1) / total_frames
        bar_w = 240
        cv2.rectangle(frame, (20, y), (20 + bar_w, y + 6), (50, 50, 60), -1)
        cv2.rectangle(frame, (20, y), (20 + int(bar_w * progress), y + 6),
                     (100, 200, 255), -1)
        
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 ПРОЦЕССОР
# ═══════════════════════════════════════════════════════════════════════════════

class SkeletonProcessor:
    """Главный процессор"""
    
    def __init__(self, model_size: str = 'n', confidence: float = 0.3,
                 style: str = 'neon'):
        self.tracker = YOLOPoseTracker(model_size, confidence)
        self.visualizer = SkeletonVisualizer(style)
        self.info_panel = InfoPanel()
    
    def process_video(self, input_path: str, output_path: str,
                      show_preview: bool = True,
                      max_frames: Optional[int] = None,
                      show_skeleton: bool = True,
                      show_trajectory: bool = True,
                      show_bbox: bool = True):
        
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
        
        print(f"\n{'═'*55}")
        print(f"  🦴 YOLO SKELETON TRACKER")
        print(f"{'═'*55}")
        print(f"  📁 Input:  {input_path}")
        print(f"  📁 Output: {output_path}")
        print(f"  📐 Size:   {width}x{height} @ {fps:.1f} FPS")
        print(f"  🧠 Model:  YOLOv8{self.tracker.model_size}-pose")
        print(f"  🎨 Style:  {self.visualizer.style}")
        print(f"{'═'*55}")
        print(f"\n  Press Q to quit, P to pause\n")
        
        frame_times = []
        frame_idx = 0
        
        while frame_idx < total:
            ret, frame = cap.read()
            if not ret:
                break
            
            start = time.time()
            
            # Трекинг
            persons = self.tracker.process_frame(frame)
            
            # Визуализация
            output_frame = self.visualizer.render(
                frame, persons,
                show_skeleton=show_skeleton,
                show_trajectory=show_trajectory,
                show_bbox=show_bbox
            )
            
            # Инфо-панель
            current_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else fps
            output_frame = self.info_panel.draw(
                output_frame, persons, frame_idx, total, current_fps
            )
            
            frame_time = time.time() - start
            frame_times.append(frame_time)
            
            out.write(output_frame)
            
            if show_preview:
                preview = cv2.resize(output_frame, (0, 0), fx=0.6, fy=0.6)
                cv2.imshow('YOLO Skeleton Tracker', preview)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  Interrupted")
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
            
            if frame_idx % 30 == 0:
                progress = (frame_idx + 1) / total * 100
                avg_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
                print(f"\r  [{progress:5.1f}%] FPS: {avg_fps:.1f} | People: {len(persons)}", end="")
            
            frame_idx += 1
        
        print(f"\n\n{'═'*55}")
        print(f"  ✅ COMPLETE")
        print(f"{'═'*55}")
        if frame_times:
            print(f"  ⏱  Avg time: {np.mean(frame_times)*1000:.1f} ms")
            print(f"  🚀 Avg FPS: {1.0/np.mean(frame_times):.1f}")
        print(f"  💾 Saved: {output_path}")
        print(f"{'═'*55}\n")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🦴 YOLO Skeleton Tracker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model sizes:
  n - nano (fastest, least accurate)
  s - small
  m - medium (recommended)
  l - large
  x - xlarge (slowest, most accurate)

Styles:
  neon    - Colorful with glow effects
  classic - Simple colored skeleton
  minimal - Thin lines

Examples:
  python yolo_skeleton_tracker.py test.mp4
  python yolo_skeleton_tracker.py test.mp4 --model m --style neon
  python yolo_skeleton_tracker.py test.mp4 --conf 0.5 --no-bbox
        """
    )
    
    parser.add_argument('input', help='Input video')
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('--model', '-m', default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--style', default='neon',
                       choices=['neon', 'classic', 'minimal'],
                       help='Visual style')
    
    # Visualization toggles
    parser.add_argument('--no-skeleton', action='store_true')
    parser.add_argument('--no-trajectory', action='store_true')
    parser.add_argument('--no-bbox', action='store_true')
    
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--max-frames', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_skeleton.mp4")
    
    processor = SkeletonProcessor(
        model_size=args.model,
        confidence=args.conf,
        style=args.style
    )
    
    processor.process_video(
        args.input,
        args.output,
        show_preview=not args.no_preview,
        max_frames=args.max_frames,
        show_skeleton=not args.no_skeleton,
        show_trajectory=not args.no_trajectory,
        show_bbox=not args.no_bbox
    )


if __name__ == '__main__':
    main()


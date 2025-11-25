#!/usr/bin/env python3
"""
🤖 SAM 2 Video Object Segmentation
===================================
Продвинутый трекинг объектов с использованием SAM 2.
Поддерживает автоматическую и интерактивную сегментацию.

Требования:
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
    
Запуск:
    python sam2_video_tracker.py video.mp4 --auto
    python sam2_video_tracker.py video.mp4 --interactive
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from collections import deque
import colorsys
import time
import tempfile
import shutil
import os


# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def generate_vibrant_colors(n: int) -> List[Tuple[int, int, int]]:
    """Генерирует яркие насыщенные цвета"""
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0  # Golden ratio
        sat = 0.85 + 0.15 * np.sin(i * 0.7)
        val = 0.95
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append(tuple(int(c * 255) for c in rgb[::-1]))  # BGR
    return colors


class SAMVisualizer:
    """Визуализатор для SAM масок"""
    
    def __init__(self, num_colors: int = 30):
        self.colors = generate_vibrant_colors(num_colors)
        
    def apply_mask(self, frame: np.ndarray, mask: np.ndarray, 
                   color: Tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
        """Применяет маску с цветом"""
        overlay = frame.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    def draw_contour(self, frame: np.ndarray, mask: np.ndarray,
                     color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
        """Рисует контур маски"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255, 255, 255), thickness + 1)
        cv2.drawContours(frame, contours, -1, color, thickness)
        return frame
    
    def draw_glow(self, frame: np.ndarray, mask: np.ndarray,
                  color: Tuple[int, int, int], blur_size: int = 25) -> np.ndarray:
        """Добавляет эффект свечения"""
        glow_mask = cv2.GaussianBlur(mask.astype(np.float32) * 255, 
                                     (blur_size, blur_size), 0)
        glow_mask = np.clip(glow_mask / 255.0 * 0.5, 0, 1)
        
        glow_layer = np.zeros_like(frame, dtype=np.float32)
        for i in range(3):
            glow_layer[:, :, i] = glow_mask * color[i]
        
        result = frame.astype(np.float32) + glow_layer
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def visualize_all(self, frame: np.ndarray, 
                      masks: Dict[int, np.ndarray],
                      trajectories: Optional[Dict[int, List]] = None) -> np.ndarray:
        """Полная визуализация всех объектов"""
        output = frame.copy()
        
        # Сначала рисуем glow эффекты
        for obj_id, mask in masks.items():
            color = self.colors[obj_id % len(self.colors)]
            output = self.draw_glow(output, mask, color)
        
        # Затем маски
        for obj_id, mask in masks.items():
            color = self.colors[obj_id % len(self.colors)]
            output = self.apply_mask(output, mask, color, alpha=0.35)
        
        # Контуры
        for obj_id, mask in masks.items():
            color = self.colors[obj_id % len(self.colors)]
            output = self.draw_contour(output, mask, color)
        
        # Траектории
        if trajectories:
            for obj_id, points in trajectories.items():
                if len(points) < 2:
                    continue
                color = self.colors[obj_id % len(self.colors)]
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    pt_color = tuple(int(c * alpha) for c in color)
                    thickness = max(1, int(3 * alpha))
                    pt1 = tuple(map(int, points[i-1]))
                    pt2 = tuple(map(int, points[i]))
                    cv2.line(output, pt1, pt2, pt_color, thickness)
        
        # Центроиды и метки
        for obj_id, mask in masks.items():
            if not mask.any():
                continue
            
            color = self.colors[obj_id % len(self.colors)]
            
            # Находим центроид
            contours, _ = cv2.findContours(mask.astype(np.uint8),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Центроид
                    cv2.circle(output, (cx, cy), 8, (255, 255, 255), 2)
                    cv2.circle(output, (cx, cy), 6, color, -1)
                    
                    # Метка
                    label = f"Object {obj_id}"
                    cv2.putText(output, label, (cx + 15, cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(output, label, (cx + 15, cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 SAM 2 WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class SAM2VideoTracker:
    """
    Обёртка над SAM 2 для video object segmentation.
    """
    
    def __init__(self, model_size: str = "large", device: str = "auto"):
        """
        Args:
            model_size: 'tiny', 'small', 'base_plus', 'large'
            device: 'cuda', 'mps', 'cpu', 'auto'
        """
        self.model_size = model_size
        self.device = self._get_device(device)
        self.predictor = None
        self.inference_state = None
        self.video_segments = {}
        self.object_ids = []
        self.visualizer = SAMVisualizer()
        self.trajectories: Dict[int, deque] = {}
        
        self._temp_dir = None
        
    def _get_device(self, device: str) -> str:
        """Определяет устройство"""
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
    
    def _load_model(self):
        """Загружает SAM 2 модель"""
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
            
            model_configs = {
                'tiny': 'sam2_hiera_t.yaml',
                'small': 'sam2_hiera_s.yaml', 
                'base_plus': 'sam2_hiera_b+.yaml',
                'large': 'sam2_hiera_l.yaml'
            }
            
            checkpoints = {
                'tiny': 'facebook/sam2-hiera-tiny',
                'small': 'facebook/sam2-hiera-small',
                'base_plus': 'facebook/sam2-hiera-base-plus',
                'large': 'facebook/sam2-hiera-large'
            }
            
            config = model_configs.get(self.model_size, 'sam2_hiera_l.yaml')
            checkpoint = checkpoints.get(self.model_size, 'facebook/sam2-hiera-large')
            
            print(f"📦 Загрузка SAM 2 ({self.model_size}) на {self.device}...")
            
            self.predictor = build_sam2_video_predictor(
                config, 
                checkpoint,
                device=self.device
            )
            
            print(f"✓ SAM 2 загружена успешно")
            return True
            
        except ImportError as e:
            print(f"❌ Ошибка импорта SAM 2: {e}")
            print("\nУстановите SAM 2:")
            print("  pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            return False
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def _extract_frames(self, video_path: str) -> str:
        """Извлекает кадры из видео во временную директорию"""
        self._temp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = os.path.join(self._temp_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
        
        cap.release()
        print(f"📁 Извлечено {frame_idx} кадров")
        return self._temp_dir
    
    def init_video(self, video_path: str):
        """Инициализирует видео для трекинга"""
        if self.predictor is None:
            if not self._load_model():
                raise RuntimeError("Не удалось загрузить SAM 2")
        
        # Извлекаем кадры
        frames_dir = self._extract_frames(video_path)
        
        # Инициализируем state
        self.inference_state = self.predictor.init_state(video_path=frames_dir)
        self.video_segments = {}
        self.object_ids = []
        
        print(f"✓ Видео инициализировано")
    
    def add_object_by_point(self, frame_idx: int, point: Tuple[int, int],
                           obj_id: Optional[int] = None) -> int:
        """
        Добавляет объект по точке.
        
        Args:
            frame_idx: Индекс кадра
            point: Точка (x, y) на объекте
            obj_id: ID объекта (если None - автоматический)
        
        Returns:
            ID добавленного объекта
        """
        if obj_id is None:
            obj_id = len(self.object_ids)
        
        points = np.array([[point]], dtype=np.float32)
        labels = np.array([[1]], dtype=np.int32)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        
        if obj_id not in self.object_ids:
            self.object_ids.append(obj_id)
            self.trajectories[obj_id] = deque(maxlen=100)
        
        print(f"✓ Добавлен объект {obj_id} по точке {point}")
        return obj_id
    
    def add_object_by_box(self, frame_idx: int, box: Tuple[int, int, int, int],
                         obj_id: Optional[int] = None) -> int:
        """
        Добавляет объект по bounding box.
        
        Args:
            frame_idx: Индекс кадра
            box: Bounding box (x1, y1, x2, y2)
            obj_id: ID объекта
        
        Returns:
            ID добавленного объекта
        """
        if obj_id is None:
            obj_id = len(self.object_ids)
        
        box_np = np.array([box], dtype=np.float32)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box_np,
        )
        
        if obj_id not in self.object_ids:
            self.object_ids.append(obj_id)
            self.trajectories[obj_id] = deque(maxlen=100)
        
        print(f"✓ Добавлен объект {obj_id} по box {box}")
        return obj_id
    
    def propagate(self) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Распространяет сегментацию на все кадры.
        
        Returns:
            Dict[frame_idx, Dict[obj_id, mask]]
        """
        print("🔄 Распространение сегментации...")
        
        self.video_segments = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in \
                self.predictor.propagate_in_video(self.inference_state):
            
            self.video_segments[out_frame_idx] = {}
            
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                self.video_segments[out_frame_idx][obj_id] = mask
                
                # Обновляем траекторию
                if mask.any():
                    ys, xs = np.where(mask)
                    centroid = (int(xs.mean()), int(ys.mean()))
                    self.trajectories[obj_id].append(centroid)
        
        print(f"✓ Обработано {len(self.video_segments)} кадров")
        return self.video_segments
    
    def process_and_save(self, input_path: str, output_path: str,
                         show_preview: bool = True):
        """
        Обрабатывает видео и сохраняет результат.
        
        Args:
            input_path: Входное видео
            output_path: Выходное видео
            show_preview: Показывать превью
        """
        cap = cv2.VideoCapture(input_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n{'='*50}")
        print("🎬 Рендеринг результата...")
        print(f"{'='*50}\n")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Получаем маски для этого кадра
            masks = self.video_segments.get(frame_idx, {})
            
            # Получаем траектории до текущего кадра
            current_trajectories = {}
            for obj_id, traj in self.trajectories.items():
                # Берем только точки до текущего кадра
                current_trajectories[obj_id] = list(traj)[:frame_idx + 1]
            
            # Визуализируем
            output_frame = self.visualizer.visualize_all(
                frame, masks, current_trajectories
            )
            
            # Добавляем информацию
            info = f"Frame: {frame_idx + 1}/{total_frames} | Objects: {len(masks)}"
            cv2.putText(output_frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(output_frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            out.write(output_frame)
            
            if show_preview:
                preview = cv2.resize(output_frame, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow('SAM 2 Tracking', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_idx % 30 == 0:
                progress = (frame_idx + 1) / total_frames * 100
                print(f"\r  Progress: {progress:.1f}%", end="")
            
            frame_idx += 1
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n\n✅ Сохранено: {output_path}")
    
    def cleanup(self):
        """Очищает временные файлы"""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 АВТОМАТИЧЕСКАЯ ДЕТЕКЦИЯ ОБЪЕКТОВ
# ═══════════════════════════════════════════════════════════════════════════════

class AutoObjectDetector:
    """
    Автоматическая детекция движущихся объектов для SAM 2.
    Использует optical flow и background subtraction для поиска объектов.
    """
    
    def __init__(self, min_area: int = 1000, max_objects: int = 10):
        self.min_area = min_area
        self.max_objects = max_objects
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=True
        )
    
    def detect_moving_objects(self, frames: List[np.ndarray], 
                              num_frames: int = 30) -> List[Dict]:
        """
        Детектирует движущиеся объекты в первых кадрах.
        
        Args:
            frames: Список кадров
            num_frames: Количество кадров для анализа
        
        Returns:
            Список обнаруженных объектов с координатами
        """
        detections_per_frame = []
        
        # Обрабатываем первые кадры
        for frame in frames[:num_frames]:
            fg_mask = self.bg_subtractor.apply(frame)
            fg_mask[fg_mask == 127] = 0  # Удаляем тени
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            frame_detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2
                
                frame_detections.append({
                    'bbox': (x, y, x + w, y + h),
                    'centroid': (cx, cy),
                    'area': area
                })
            
            detections_per_frame.append(frame_detections)
        
        # Выбираем стабильные объекты (присутствуют в нескольких кадрах)
        stable_objects = self._find_stable_objects(detections_per_frame)
        
        return stable_objects[:self.max_objects]
    
    def _find_stable_objects(self, detections_per_frame: List[List[Dict]],
                             min_frames: int = 5, max_distance: int = 100) -> List[Dict]:
        """Находит объекты, стабильно присутствующие в нескольких кадрах"""
        if not detections_per_frame:
            return []
        
        # Начинаем с детекций в середине последовательности
        mid_idx = len(detections_per_frame) // 2
        if not detections_per_frame[mid_idx]:
            return []
        
        stable = []
        
        for det in detections_per_frame[mid_idx]:
            count = 1
            total_cx, total_cy = det['centroid']
            
            # Проверяем наличие в других кадрах
            for i, frame_dets in enumerate(detections_per_frame):
                if i == mid_idx:
                    continue
                
                for other_det in frame_dets:
                    dist = np.sqrt(
                        (det['centroid'][0] - other_det['centroid'][0])**2 +
                        (det['centroid'][1] - other_det['centroid'][1])**2
                    )
                    if dist < max_distance:
                        count += 1
                        total_cx += other_det['centroid'][0]
                        total_cy += other_det['centroid'][1]
                        break
            
            if count >= min_frames:
                avg_centroid = (total_cx // count, total_cy // count)
                stable.append({
                    'centroid': avg_centroid,
                    'bbox': det['bbox'],
                    'stability': count
                })
        
        # Сортируем по стабильности
        stable.sort(key=lambda x: x['stability'], reverse=True)
        
        return stable


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 ИНТЕРАКТИВНЫЙ РЕЖИМ
# ═══════════════════════════════════════════════════════════════════════════════

class InteractiveSAM2:
    """Интерактивный режим выбора объектов для SAM 2"""
    
    def __init__(self, tracker: SAM2VideoTracker):
        self.tracker = tracker
        self.current_frame = None
        self.current_frame_idx = 0
        self.selected_points = []
        self.preview_mask = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Обработчик мыши"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append((x, y))
            print(f"  Point added: ({x}, {y})")
    
    def select_objects(self, video_path: str) -> List[Dict]:
        """
        Интерактивный выбор объектов.
        
        Returns:
            Список выбранных объектов с точками
        """
        cap = cv2.VideoCapture(video_path)
        ret, self.current_frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Не удалось прочитать видео")
            return []
        
        window_name = "Select Objects (LMB=add, N=next object, ENTER=done, Q=quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n" + "="*50)
        print("🎯 ИНТЕРАКТИВНЫЙ ВЫБОР ОБЪЕКТОВ")
        print("="*50)
        print("  LMB   - добавить точку объекта")
        print("  N     - следующий объект")
        print("  D     - удалить последнюю точку")
        print("  ENTER - завершить выбор")
        print("  Q     - отмена")
        print("="*50 + "\n")
        
        objects = []
        current_object_points = []
        
        while True:
            display = self.current_frame.copy()
            
            # Рисуем уже выбранные объекты
            for i, obj in enumerate(objects):
                color = self.tracker.visualizer.colors[i % len(self.tracker.visualizer.colors)]
                for pt in obj['points']:
                    cv2.circle(display, pt, 8, (255, 255, 255), 2)
                    cv2.circle(display, pt, 6, color, -1)
            
            # Рисуем текущие точки
            color = self.tracker.visualizer.colors[len(objects) % len(self.tracker.visualizer.colors)]
            for pt in self.selected_points:
                cv2.circle(display, pt, 8, (255, 255, 255), 2)
                cv2.circle(display, pt, 6, color, -1)
            
            # Информация
            info = f"Objects: {len(objects)} | Current points: {len(self.selected_points)}"
            cv2.putText(display, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(display, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("  Отмена")
                cv2.destroyAllWindows()
                return []
            
            elif key == 13:  # ENTER
                if self.selected_points:
                    objects.append({'points': self.selected_points.copy()})
                    self.selected_points = []
                break
            
            elif key == ord('n'):
                if self.selected_points:
                    objects.append({'points': self.selected_points.copy()})
                    self.selected_points = []
                    print(f"  Object {len(objects)} saved")
            
            elif key == ord('d'):
                if self.selected_points:
                    self.selected_points.pop()
                    print("  Last point removed")
        
        cv2.destroyAllWindows()
        print(f"\n✓ Выбрано объектов: {len(objects)}")
        
        return objects


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🤖 SAM 2 Video Object Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input', type=str, help='Input video path')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-detect moving objects')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive object selection')
    parser.add_argument('--model', type=str, default='large',
                       choices=['tiny', 'small', 'base_plus', 'large'],
                       help='SAM 2 model size')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cuda', 'mps', 'cpu', 'auto'],
                       help='Device to use')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable preview')
    
    args = parser.parse_args()
    
    # Выходной путь
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_sam2.mp4")
    
    print(f"\n{'═'*60}")
    print(f"  🤖 SAM 2 VIDEO OBJECT SEGMENTATION")
    print(f"{'═'*60}")
    print(f"  📁 Input:  {args.input}")
    print(f"  📁 Output: {args.output}")
    print(f"  🧠 Model:  {args.model}")
    print(f"  💻 Device: {args.device}")
    print(f"{'═'*60}\n")
    
    # Создаем трекер
    tracker = SAM2VideoTracker(model_size=args.model, device=args.device)
    
    try:
        # Инициализируем видео
        tracker.init_video(args.input)
        
        if args.interactive:
            # Интерактивный режим
            interactive = InteractiveSAM2(tracker)
            objects = interactive.select_objects(args.input)
            
            if not objects:
                print("❌ Объекты не выбраны")
                return
            
            # Добавляем объекты
            for i, obj in enumerate(objects):
                # Используем первую точку каждого объекта
                point = obj['points'][0]
                tracker.add_object_by_point(0, point, obj_id=i)
        
        elif args.auto:
            # Автоматический режим
            print("🔍 Автоматическая детекция объектов...")
            
            # Читаем первые кадры
            cap = cv2.VideoCapture(args.input)
            frames = []
            for _ in range(50):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            # Детектируем объекты
            detector = AutoObjectDetector(min_area=1500, max_objects=8)
            detected = detector.detect_moving_objects(frames)
            
            if not detected:
                print("⚠ Движущиеся объекты не обнаружены")
                print("  Попробуйте --interactive режим")
                return
            
            print(f"✓ Обнаружено объектов: {len(detected)}")
            
            # Добавляем объекты в SAM 2
            for i, obj in enumerate(detected):
                tracker.add_object_by_point(0, obj['centroid'], obj_id=i)
        
        else:
            print("⚠ Укажите режим: --auto или --interactive")
            return
        
        # Распространяем сегментацию
        tracker.propagate()
        
        # Сохраняем результат
        tracker.process_and_save(args.input, args.output, 
                                show_preview=not args.no_preview)
        
    finally:
        tracker.cleanup()


if __name__ == '__main__':
    main()


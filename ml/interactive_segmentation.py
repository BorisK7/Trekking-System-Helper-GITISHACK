"""
🎨 Interactive Video Object Segmentation
=========================================
Интерактивная сегментация видео с выбором объектов мышкой.
Поддерживает SAM 2 и fallback на OpenCV.

Фичи:
- Кликните на объект чтобы начать его отслеживать
- ПКМ для удаления точек
- Красивая визуализация с градиентами и эффектами
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from collections import deque
import colorsys
import time


# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 ПРОДВИНУТАЯ ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

class GlowEffect:
    """Создает эффект свечения"""
    
    @staticmethod
    def apply(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int],
              intensity: float = 0.8, blur_size: int = 21) -> np.ndarray:
        """Применяет эффект свечения к маске"""
        # Создаем размытую версию маски
        glow_mask = cv2.GaussianBlur(mask.astype(np.float32), 
                                     (blur_size, blur_size), 0)
        glow_mask = np.clip(glow_mask * intensity, 0, 1)
        
        # Создаем цветной слой
        glow_layer = np.zeros_like(image, dtype=np.float32)
        for i, c in enumerate(color):
            glow_layer[:, :, i] = glow_mask * c
        
        # Смешиваем
        result = cv2.addWeighted(image.astype(np.float32), 1.0,
                                glow_layer, 0.5, 0)
        return np.clip(result, 0, 255).astype(np.uint8)


class GradientMask:
    """Создает градиентные маски"""
    
    @staticmethod
    def create_radial(center: Tuple[int, int], radius: int,
                      shape: Tuple[int, int]) -> np.ndarray:
        """Создает радиальный градиент"""
        y, x = np.ogrid[:shape[0], :shape[1]]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        gradient = np.clip(1 - dist / radius, 0, 1)
        return gradient
    
    @staticmethod
    def apply_gradient_fill(image: np.ndarray, mask: np.ndarray,
                           color1: Tuple[int, int, int],
                           color2: Tuple[int, int, int],
                           direction: str = 'vertical') -> np.ndarray:
        """Заливает маску градиентом"""
        h, w = mask.shape
        
        if direction == 'vertical':
            gradient = np.linspace(0, 1, h).reshape(h, 1)
        elif direction == 'horizontal':
            gradient = np.linspace(0, 1, w).reshape(1, w)
        else:  # diagonal
            gradient = (np.linspace(0, 1, h).reshape(h, 1) + 
                       np.linspace(0, 1, w).reshape(1, w)) / 2
        
        gradient = np.broadcast_to(gradient, (h, w))
        
        # Интерполируем цвета
        color_layer = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(3):
            color_layer[:, :, i] = color1[i] * (1 - gradient) + color2[i] * gradient
        
        # Применяем маску
        result = image.copy().astype(np.float32)
        mask_3d = np.stack([mask] * 3, axis=-1).astype(np.float32)
        
        result = result * (1 - mask_3d * 0.5) + color_layer * mask_3d * 0.5
        return np.clip(result, 0, 255).astype(np.uint8)


class ParticleSystem:
    """Система частиц для визуальных эффектов"""
    
    def __init__(self, max_particles: int = 100):
        self.particles = []
        self.max_particles = max_particles
    
    def emit(self, position: Tuple[int, int], color: Tuple[int, int, int],
             velocity: Tuple[float, float] = (0, 0), count: int = 5):
        """Испускает частицы"""
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                self.particles.pop(0)
            
            particle = {
                'pos': list(position),
                'vel': [velocity[0] + np.random.randn() * 2,
                       velocity[1] + np.random.randn() * 2],
                'color': color,
                'life': 1.0,
                'size': np.random.randint(2, 6)
            }
            self.particles.append(particle)
    
    def update(self, dt: float = 0.033):
        """Обновляет частицы"""
        alive = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.5  # Гравитация
            p['life'] -= dt * 2
            
            if p['life'] > 0:
                alive.append(p)
        
        self.particles = alive
    
    def render(self, frame: np.ndarray) -> np.ndarray:
        """Рендерит частицы"""
        for p in self.particles:
            alpha = max(0, p['life'])
            color = tuple(int(c * alpha) for c in p['color'])
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['size'] * alpha)
            
            if size > 0:
                cv2.circle(frame, pos, size, color, -1)
        
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 ИНТЕРАКТИВНЫЙ ТРЕКЕР
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SegmentedObject:
    """Сегментированный объект"""
    id: int
    color: Tuple[int, int, int]
    mask: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    centroid: Optional[Tuple[int, int]] = None
    trajectory: deque = field(default_factory=lambda: deque(maxlen=100))
    prompt_points: List[Tuple[int, int]] = field(default_factory=list)
    prompt_labels: List[int] = field(default_factory=list)  # 1=foreground, 0=background
    active: bool = True
    
    def add_point(self, point: Tuple[int, int], is_foreground: bool = True):
        """Добавляет точку промпта"""
        self.prompt_points.append(point)
        self.prompt_labels.append(1 if is_foreground else 0)
    
    def remove_last_point(self):
        """Удаляет последнюю точку"""
        if self.prompt_points:
            self.prompt_points.pop()
            self.prompt_labels.pop()


class InteractiveSegmenter:
    """
    Интерактивная сегментация с выбором объектов мышкой.
    Использует OpenCV для базовой сегментации (GrabCut) или SAM 2 если доступен.
    """
    
    def __init__(self, use_sam: bool = False):
        self.use_sam = use_sam
        self.objects: Dict[int, SegmentedObject] = {}
        self.next_id = 0
        self.current_object_id: Optional[int] = None
        self.colors = self._generate_palette(30)
        
        # SAM модель (если используется)
        self.sam_predictor = None
        
        # UI состояние
        self.mouse_pos = (0, 0)
        self.drawing_mode = 'point'  # 'point', 'box'
        self.box_start = None
        
        # Эффекты
        self.particles = ParticleSystem()
        self.show_particles = True
        
    def _generate_palette(self, n: int) -> List[Tuple[int, int, int]]:
        """Генерирует яркую палитру"""
        colors = []
        golden_ratio = 0.618033988749895
        h = np.random.random()
        
        for _ in range(n):
            h = (h + golden_ratio) % 1
            # Используем яркие насыщенные цвета
            rgb = colorsys.hsv_to_rgb(h, 0.85, 0.95)
            colors.append(tuple(int(c * 255) for c in rgb))
        
        return colors
    
    def _init_sam(self):
        """Инициализирует SAM модель"""
        if self.sam_predictor is not None:
            return True
        
        try:
            # Пробуем SAM 2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            sam2_model = build_sam2("sam2_hiera_l.yaml", "facebook/sam2-hiera-large")
            self.sam_predictor = SAM2ImagePredictor(sam2_model)
            print("✓ SAM 2 загружена")
            return True
            
        except ImportError:
            try:
                # Пробуем оригинальный SAM
                from segment_anything import sam_model_registry, SamPredictor
                
                # Путь к весам (нужно скачать)
                sam = sam_model_registry["vit_h"]("sam_vit_h_4b8939.pth")
                self.sam_predictor = SamPredictor(sam)
                print("✓ SAM загружена")
                return True
                
            except (ImportError, FileNotFoundError):
                print("⚠ SAM не доступна, используется GrabCut")
                return False
    
    def create_object(self) -> int:
        """Создает новый объект для отслеживания"""
        obj_id = self.next_id
        color = self.colors[obj_id % len(self.colors)]
        
        self.objects[obj_id] = SegmentedObject(id=obj_id, color=color)
        self.current_object_id = obj_id
        self.next_id += 1
        
        return obj_id
    
    def segment_with_points(self, frame: np.ndarray, 
                           obj: SegmentedObject) -> np.ndarray:
        """Сегментирует объект по точкам промпта"""
        if not obj.prompt_points:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        
        if self.use_sam and self.sam_predictor is not None:
            return self._segment_sam(frame, obj)
        else:
            return self._segment_grabcut(frame, obj)
    
    def _segment_sam(self, frame: np.ndarray, 
                     obj: SegmentedObject) -> np.ndarray:
        """Сегментация с SAM"""
        self.sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        points = np.array(obj.prompt_points)
        labels = np.array(obj.prompt_labels)
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )
        
        # Берем маску с лучшим скором
        best_mask = masks[np.argmax(scores)]
        return (best_mask * 255).astype(np.uint8)
    
    def _segment_grabcut(self, frame: np.ndarray,
                         obj: SegmentedObject) -> np.ndarray:
        """Сегментация с GrabCut (fallback)"""
        h, w = frame.shape[:2]
        
        # Создаем начальную маску на основе точек
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Foreground точки
        fg_points = [p for p, l in zip(obj.prompt_points, obj.prompt_labels) if l == 1]
        bg_points = [p for p, l in zip(obj.prompt_points, obj.prompt_labels) if l == 0]
        
        if not fg_points:
            return mask
        
        # Оцениваем bbox по foreground точкам
        fg_arr = np.array(fg_points)
        x_min, y_min = fg_arr.min(axis=0)
        x_max, y_max = fg_arr.max(axis=0)
        
        # Расширяем bbox
        padding = 50
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        rect = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Инициализируем GrabCut маску
        gc_mask = np.zeros((h, w), dtype=np.uint8)
        gc_mask[:] = cv2.GC_BGD  # Background
        gc_mask[y_min:y_max, x_min:x_max] = cv2.GC_PR_BGD  # Probable background
        
        # Отмечаем точки
        for p in fg_points:
            cv2.circle(gc_mask, tuple(map(int, p)), 10, cv2.GC_FGD, -1)
        for p in bg_points:
            cv2.circle(gc_mask, tuple(map(int, p)), 10, cv2.GC_BGD, -1)
        
        # Запускаем GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(frame, gc_mask, rect, bgd_model, fgd_model, 
                       5, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            # Если GrabCut не сработал, создаем простую маску
            for p in fg_points:
                cv2.circle(mask, tuple(map(int, p)), 30, 255, -1)
            return mask
        
        # Извлекаем маску
        mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                       255, 0).astype(np.uint8)
        
        return mask
    
    def update_object_mask(self, frame: np.ndarray, obj_id: int):
        """Обновляет маску объекта"""
        if obj_id not in self.objects:
            return
        
        obj = self.objects[obj_id]
        obj.mask = self.segment_with_points(frame, obj)
        
        # Вычисляем bbox и центроид
        if obj.mask is not None and obj.mask.any():
            contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Берем самый большой контур
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                obj.bbox = (x, y, w, h)
                
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    obj.centroid = (cx, cy)
                    obj.trajectory.append((cx, cy))
    
    def render(self, frame: np.ndarray, show_prompts: bool = True) -> np.ndarray:
        """Рендерит визуализацию"""
        output = frame.copy()
        
        # Рендерим маски объектов
        for obj_id, obj in self.objects.items():
            if not obj.active or obj.mask is None:
                continue
            
            # Применяем маску с градиентом
            output = GradientMask.apply_gradient_fill(
                output, obj.mask / 255.0,
                obj.color,
                tuple(min(255, int(c * 1.3)) for c in obj.color),
                'diagonal'
            )
            
            # Glow эффект
            output = GlowEffect.apply(output, obj.mask / 255.0, obj.color,
                                     intensity=0.4, blur_size=31)
            
            # Контур
            contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (255, 255, 255), 2)
            cv2.drawContours(output, contours, -1, obj.color, 1)
            
            # Траектория
            if len(obj.trajectory) > 1:
                points = list(obj.trajectory)
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    color = tuple(int(c * alpha) for c in obj.color)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(output, 
                            tuple(map(int, points[i-1])),
                            tuple(map(int, points[i])),
                            color, thickness)
            
            # Центроид
            if obj.centroid:
                cv2.circle(output, obj.centroid, 8, (255, 255, 255), 2)
                cv2.circle(output, obj.centroid, 6, obj.color, -1)
            
            # Метка
            if obj.bbox:
                x, y, w, h = obj.bbox
                label = f"Object {obj.id}"
                cv2.putText(output, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(output, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, obj.color, 1)
        
        # Рендерим точки промптов
        if show_prompts:
            for obj_id, obj in self.objects.items():
                for point, label in zip(obj.prompt_points, obj.prompt_labels):
                    color = (0, 255, 0) if label == 1 else (0, 0, 255)
                    cv2.circle(output, tuple(map(int, point)), 8, (255, 255, 255), 2)
                    cv2.circle(output, tuple(map(int, point)), 6, color, -1)
        
        # Рендерим частицы
        if self.show_particles:
            self.particles.update()
            output = self.particles.render(output)
        
        return output
    
    def render_ui(self, frame: np.ndarray) -> np.ndarray:
        """Рендерит UI элементы"""
        h, w = frame.shape[:2]
        
        # Подсказки
        hints = [
            "LMB: Add foreground point",
            "RMB: Add background point",
            "N: New object",
            "D: Delete last point",
            "R: Reset current object",
            "S: Save mask",
            "Q: Quit"
        ]
        
        y = 30
        for hint in hints:
            cv2.putText(frame, hint, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, hint, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20
        
        # Текущий объект
        if self.current_object_id is not None:
            obj = self.objects.get(self.current_object_id)
            if obj:
                status = f"Current: Object {obj.id} | Points: {len(obj.prompt_points)}"
                cv2.putText(frame, status, (10, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(frame, status, (10, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, obj.color, 2)
        
        # Курсор
        cv2.circle(frame, self.mouse_pos, 10, (255, 255, 255), 1)
        cv2.circle(frame, self.mouse_pos, 2, (255, 255, 255), -1)
        
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 ИНТЕРАКТИВНОЕ ПРИЛОЖЕНИЕ
# ═══════════════════════════════════════════════════════════════════════════════

class InteractiveApp:
    """Интерактивное приложение для сегментации"""
    
    def __init__(self, video_path: str, use_sam: bool = False):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.current_frame_idx = 0
        self.current_frame = None
        
        self.segmenter = InteractiveSegmenter(use_sam=use_sam)
        
        # Пробуем инициализировать SAM
        if use_sam:
            self.segmenter._init_sam()
        
        # Окно
        self.window_name = "Interactive Segmentation"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Читаем первый кадр
        self._read_frame()
    
    def _read_frame(self):
        """Читает текущий кадр"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
        return ret
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Обработчик событий мыши"""
        self.segmenter.mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Добавляем foreground точку
            if self.segmenter.current_object_id is None:
                self.segmenter.create_object()
            
            obj = self.segmenter.objects[self.segmenter.current_object_id]
            obj.add_point((x, y), is_foreground=True)
            
            # Частицы
            self.segmenter.particles.emit((x, y), obj.color, count=10)
            
            # Обновляем маску
            if self.current_frame is not None:
                self.segmenter.update_object_mask(
                    self.current_frame, 
                    self.segmenter.current_object_id
                )
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Добавляем background точку
            if self.segmenter.current_object_id is not None:
                obj = self.segmenter.objects[self.segmenter.current_object_id]
                obj.add_point((x, y), is_foreground=False)
                
                # Обновляем маску
                if self.current_frame is not None:
                    self.segmenter.update_object_mask(
                        self.current_frame,
                        self.segmenter.current_object_id
                    )
    
    def run(self):
        """Запускает приложение"""
        print(f"\n{'='*60}")
        print("🎨 INTERACTIVE SEGMENTATION")
        print(f"{'='*60}")
        print(f"📁 Video: {self.video_path}")
        print(f"📐 Resolution: {self.width}x{self.height}")
        print(f"🎞  FPS: {self.fps}")
        print(f"📊 Frames: {self.total_frames}")
        print(f"{'='*60}\n")
        
        print("Controls:")
        print("  LMB - Add foreground point")
        print("  RMB - Add background point")
        print("  N   - New object")
        print("  D   - Delete last point")
        print("  R   - Reset current object")
        print("  Space - Play/Pause")
        print("  ←/→ - Previous/Next frame")
        print("  S   - Save output")
        print("  Q   - Quit\n")
        
        playing = False
        
        while True:
            if self.current_frame is None:
                break
            
            # Рендерим
            display = self.segmenter.render(self.current_frame)
            display = self.segmenter.render_ui(display)
            
            # Показываем номер кадра
            frame_info = f"Frame: {self.current_frame_idx + 1}/{self.total_frames}"
            cv2.putText(display, frame_info, (self.width - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(self.window_name, display)
            
            # Обработка клавиш
            key = cv2.waitKey(30 if playing else 0) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('n'):
                # Новый объект
                obj_id = self.segmenter.create_object()
                print(f"Created new object: {obj_id}")
            
            elif key == ord('d'):
                # Удалить последнюю точку
                if self.segmenter.current_object_id is not None:
                    obj = self.segmenter.objects[self.segmenter.current_object_id]
                    obj.remove_last_point()
                    self.segmenter.update_object_mask(
                        self.current_frame,
                        self.segmenter.current_object_id
                    )
            
            elif key == ord('r'):
                # Сбросить текущий объект
                if self.segmenter.current_object_id is not None:
                    obj = self.segmenter.objects[self.segmenter.current_object_id]
                    obj.prompt_points.clear()
                    obj.prompt_labels.clear()
                    obj.mask = None
                    obj.trajectory.clear()
            
            elif key == ord(' '):
                # Play/Pause
                playing = not playing
            
            elif key == 83 or key == ord('d'):  # Right arrow
                # Следующий кадр
                self.current_frame_idx = min(self.current_frame_idx + 1, 
                                            self.total_frames - 1)
                self._read_frame()
                # Обновляем маски для всех объектов
                for obj_id in self.segmenter.objects:
                    self.segmenter.update_object_mask(self.current_frame, obj_id)
            
            elif key == 81 or key == ord('a'):  # Left arrow
                # Предыдущий кадр
                self.current_frame_idx = max(self.current_frame_idx - 1, 0)
                self._read_frame()
                # Обновляем маски для всех объектов
                for obj_id in self.segmenter.objects:
                    self.segmenter.update_object_mask(self.current_frame, obj_id)
            
            elif key == ord('s'):
                # Сохранить
                output_path = Path(self.video_path).parent / "segmentation_output.png"
                cv2.imwrite(str(output_path), display)
                print(f"Saved: {output_path}")
            
            # Автовоспроизведение
            if playing:
                self.current_frame_idx += 1
                if self.current_frame_idx >= self.total_frames:
                    self.current_frame_idx = 0
                    playing = False
                self._read_frame()
        
        self.cap.release()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='🎨 Interactive Video Segmentation')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--sam', action='store_true', help='Use SAM for segmentation')
    
    args = parser.parse_args()
    
    app = InteractiveApp(args.video, use_sam=args.sam)
    app.run()


if __name__ == '__main__':
    main()


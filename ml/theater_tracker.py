#!/usr/bin/env python3
"""
🎭 Theater Scene Object Tracker
================================
Специализированный трекер для театральных сцен.
Фильтрует проблемы освещения: прожекторы, тени, блики.

Запуск:
    python theater_tracker.py test.mp4
    python theater_tracker.py test.mp4 --lighting-filter strong
    python theater_tracker.py test.mp4 --mask-lights
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
# 💡 ФИЛЬТРАЦИЯ ОСВЕЩЕНИЯ
# ═══════════════════════════════════════════════════════════════════════════════

class LightingFilter:
    """
    Фильтрация театрального освещения.
    Решает проблемы: прожекторы, софиты, блики, тени, изменение цвета света.
    """
    
    def __init__(self, mode: str = 'adaptive'):
        """
        Args:
            mode: 'light', 'medium', 'strong', 'adaptive'
        """
        self.mode = mode
        
        # CLAHE для адаптивной нормализации
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Параметры для разных режимов
        self.params = {
            'light': {
                'clahe_clip': 1.5,
                'brightness_limit': 250,
                'shadow_boost': 1.2,
                'highlight_reduce': 0.9
            },
            'medium': {
                'clahe_clip': 2.5,
                'brightness_limit': 240,
                'shadow_boost': 1.4,
                'highlight_reduce': 0.8
            },
            'strong': {
                'clahe_clip': 3.5,
                'brightness_limit': 220,
                'shadow_boost': 1.6,
                'highlight_reduce': 0.7
            },
            'adaptive': {
                'clahe_clip': 2.0,
                'brightness_limit': 235,
                'shadow_boost': 1.3,
                'highlight_reduce': 0.85
            }
        }
        
        self._update_clahe()
        
        # История для адаптивной фильтрации
        self.brightness_history = deque(maxlen=30)
        self.light_positions = []  # Обнаруженные позиции прожекторов
        
    def _update_clahe(self):
        """Обновляет CLAHE с текущими параметрами"""
        params = self.params.get(self.mode, self.params['adaptive'])
        self.clahe = cv2.createCLAHE(
            clipLimit=params['clahe_clip'], 
            tileGridSize=(8, 8)
        )
    
    def normalize_lighting(self, frame: np.ndarray) -> np.ndarray:
        """
        Основная функция нормализации освещения.
        
        Returns:
            Кадр с нормализованным освещением
        """
        # 1. Конвертируем в LAB для работы с яркостью отдельно
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # 2. Применяем CLAHE к L-каналу
        l_normalized = self.clahe.apply(l_channel)
        
        # 3. Собираем обратно
        lab_normalized = cv2.merge([l_normalized, a_channel, b_channel])
        result = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
        
        return result
    
    def detect_spotlights(self, frame: np.ndarray, 
                          threshold: int = 245) -> np.ndarray:
        """
        Обнаруживает области прожекторов (очень яркие зоны).
        
        Returns:
            Маска прожекторов (255 = прожектор)
        """
        # Конвертируем в grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Порог для очень ярких областей
        _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Расширяем маску (прожекторы имеют ореол)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bright_mask = cv2.dilate(bright_mask, kernel, iterations=2)
        
        return bright_mask
    
    def reduce_highlights(self, frame: np.ndarray, 
                          strength: float = 0.8) -> np.ndarray:
        """
        Уменьшает яркие области (пересветы).
        """
        # Конвертируем в HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Находим яркие пиксели
        v_channel = hsv[:, :, 2]
        bright_mask = v_channel > 200
        
        # Уменьшаем яркость ярких областей
        v_channel[bright_mask] = 200 + (v_channel[bright_mask] - 200) * strength
        
        hsv[:, :, 2] = np.clip(v_channel, 0, 255)
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result
    
    def boost_shadows(self, frame: np.ndarray, 
                      strength: float = 1.3) -> np.ndarray:
        """
        Поднимает тени (тёмные области).
        """
        # Конвертируем в LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        l_channel = lab[:, :, 0]
        
        # Находим тёмные пиксели
        dark_mask = l_channel < 80
        
        # Поднимаем яркость тёмных областей
        l_channel[dark_mask] = l_channel[dark_mask] * strength
        
        lab[:, :, 0] = np.clip(l_channel, 0, 255)
        
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result
    
    def homomorphic_filter(self, frame: np.ndarray, 
                           gamma_l: float = 0.5, 
                           gamma_h: float = 1.5,
                           cutoff: float = 30) -> np.ndarray:
        """
        Гомоморфная фильтрация - отделяет освещение от отражения.
        Очень эффективна для неравномерного освещения.
        """
        # Работаем с каждым каналом отдельно
        result = np.zeros_like(frame, dtype=np.float32)
        
        for i in range(3):
            channel = frame[:, :, i].astype(np.float32)
            
            # Добавляем 1 чтобы избежать log(0)
            channel = np.log1p(channel)
            
            # FFT
            dft = np.fft.fft2(channel)
            dft_shift = np.fft.fftshift(dft)
            
            # Создаем высокочастотный фильтр
            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2
            
            # Гауссов фильтр
            x = np.arange(cols) - ccol
            y = np.arange(rows) - crow
            X, Y = np.meshgrid(x, y)
            D = np.sqrt(X**2 + Y**2)
            
            # H(u,v) = (gamma_h - gamma_l) * (1 - exp(-c*(D^2/D0^2))) + gamma_l
            H = (gamma_h - gamma_l) * (1 - np.exp(-0.5 * (D**2 / cutoff**2))) + gamma_l
            
            # Применяем фильтр
            filtered = dft_shift * H
            
            # Обратное FFT
            f_ishift = np.fft.ifftshift(filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.real(img_back)
            
            # Обратный логарифм
            result[:, :, i] = np.expm1(img_back)
        
        # Нормализуем
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)
    
    def retinex_filter(self, frame: np.ndarray, 
                       sigma_list: List[int] = [15, 80, 250]) -> np.ndarray:
        """
        Multi-Scale Retinex - имитирует адаптацию глаза к освещению.
        Отлично работает для театрального освещения.
        """
        frame_float = frame.astype(np.float32) + 1.0  # Избегаем log(0)
        
        retinex = np.zeros_like(frame_float)
        
        for sigma in sigma_list:
            # Размытие (оценка освещения)
            blur = cv2.GaussianBlur(frame_float, (0, 0), sigma)
            
            # Retinex: log(I) - log(L)
            retinex += np.log10(frame_float) - np.log10(blur + 1.0)
        
        retinex = retinex / len(sigma_list)
        
        # Нормализуем в диапазон [0, 255]
        for i in range(3):
            channel = retinex[:, :, i]
            min_val = np.percentile(channel, 1)
            max_val = np.percentile(channel, 99)
            retinex[:, :, i] = np.clip((channel - min_val) / (max_val - min_val) * 255, 0, 255)
        
        return retinex.astype(np.uint8)
    
    def adaptive_filter(self, frame: np.ndarray) -> np.ndarray:
        """
        Адаптивная фильтрация - анализирует кадр и выбирает лучший метод.
        """
        # Анализируем текущий кадр
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        self.brightness_history.append(mean_brightness)
        
        # Определяем характер освещения
        high_contrast = std_brightness > 60
        very_bright = mean_brightness > 180
        very_dark = mean_brightness < 60
        uneven = std_brightness > 50 and mean_brightness > 80
        
        result = frame.copy()
        
        # Выбираем обработку
        if very_bright:
            # Много света - уменьшаем highlights
            result = self.reduce_highlights(result, 0.7)
            result = self.normalize_lighting(result)
        
        elif very_dark:
            # Мало света - поднимаем shadows
            result = self.boost_shadows(result, 1.5)
            result = self.normalize_lighting(result)
        
        elif uneven:
            # Неравномерное освещение - retinex
            result = self.retinex_filter(result)
        
        elif high_contrast:
            # Высокий контраст - гомоморфный фильтр
            result = self.homomorphic_filter(result)
        
        else:
            # Обычное освещение - просто CLAHE
            result = self.normalize_lighting(result)
        
        return result
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Главная функция обработки.
        """
        if self.mode == 'adaptive':
            return self.adaptive_filter(frame)
        
        params = self.params[self.mode]
        
        # Последовательная обработка
        result = frame.copy()
        
        # 1. Уменьшаем яркие области
        result = self.reduce_highlights(result, params['highlight_reduce'])
        
        # 2. Поднимаем тени
        result = self.boost_shadows(result, params['shadow_boost'])
        
        # 3. Нормализуем освещение
        result = self.normalize_lighting(result)
        
        return result
    
    def create_spotlight_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Создает маску для исключения прожекторов из трекинга.
        """
        # Обнаруживаем яркие области
        spotlight_mask = self.detect_spotlights(frame, threshold=240)
        
        # Инвертируем - 255 = область для трекинга, 0 = прожектор
        tracking_mask = cv2.bitwise_not(spotlight_mask)
        
        return tracking_mask


# ═══════════════════════════════════════════════════════════════════════════════
# 🎭 ТЕАТРАЛЬНЫЙ ТРЕКЕР
# ═══════════════════════════════════════════════════════════════════════════════

class TheaterObjectTracker:
    """
    Трекер объектов, оптимизированный для театральных сцен.
    """
    
    def __init__(self, 
                 lighting_filter: str = 'adaptive',
                 mask_spotlights: bool = True,
                 min_area: int = 1500,
                 max_objects: int = 15):
        
        self.lighting_filter = LightingFilter(mode=lighting_filter)
        self.mask_spotlights = mask_spotlights
        self.min_area = min_area
        self.max_objects = max_objects
        
        # Background subtractor с параметрами для театра
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=35,  # Ниже порог для лучшей детекции
            detectShadows=True
        )
        
        # KNN как альтернатива (лучше работает с тенями)
        self.bg_subtractor_knn = cv2.createBackgroundSubtractorKNN(
            history=300,
            dist2Threshold=400,
            detectShadows=True
        )
        
        self.objects: Dict[int, 'TrackedObject'] = {}
        self.next_id = 0
        self.colors = self._generate_palette(30)
        
        self.use_knn = False  # Переключатель между MOG2 и KNN
        
    def _generate_palette(self, n: int) -> List[Tuple[int, int, int]]:
        """Генерирует яркую палитру"""
        colors = []
        for i in range(n):
            hue = (i * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
            colors.append(tuple(int(c * 255) for c in rgb[::-1]))
        return colors
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List['TrackedObject'], np.ndarray]:
        """
        Обрабатывает кадр.
        
        Returns:
            (список объектов, обработанный кадр)
        """
        # 1. Фильтруем освещение
        filtered = self.lighting_filter.process(frame)
        
        # 2. Получаем маску прожекторов (если включено)
        if self.mask_spotlights:
            spotlight_mask = self.lighting_filter.create_spotlight_mask(frame)
        else:
            spotlight_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        # 3. Background subtraction
        if self.use_knn:
            fg_mask = self.bg_subtractor_knn.apply(filtered)
        else:
            fg_mask = self.bg_subtractor.apply(filtered)
        
        # 4. Удаляем тени
        fg_mask[fg_mask == 127] = 0
        
        # 5. Применяем маску прожекторов
        fg_mask = cv2.bitwise_and(fg_mask, spotlight_mask)
        
        # 6. Морфология
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 7. Находим контуры
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # 8. Извлекаем детекции
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Фильтруем по соотношению сторон
            aspect = w / h if h > 0 else 0
            if aspect < 0.15 or aspect > 6:
                continue
            
            # Центроид
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # Маска объекта
            obj_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(obj_mask, [contour], -1, 255, -1)
            
            detections.append({
                'bbox': (x, y, w, h),
                'centroid': (cx, cy),
                'mask': obj_mask,
                'area': area
            })
        
        # 9. Сортируем по площади
        detections.sort(key=lambda d: d['area'], reverse=True)
        detections = detections[:self.max_objects]
        
        # 10. Сопоставляем с объектами
        self._match_detections(detections)
        
        return list(self.objects.values()), filtered
    
    def _match_detections(self, detections: List[Dict]):
        """Сопоставляет детекции с объектами"""
        for obj in self.objects.values():
            obj.lost_frames += 1
        
        if not detections:
            self._cleanup_lost()
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
                
                # Расстояние до центроида
                dist = np.sqrt(
                    (obj.centroid[0] - det['centroid'][0])**2 +
                    (obj.centroid[1] - det['centroid'][1])**2
                )
                
                # Учитываем предсказание по скорости
                vel = obj.get_velocity()
                pred = (obj.centroid[0] + vel[0], obj.centroid[1] + vel[1])
                pred_dist = np.sqrt(
                    (pred[0] - det['centroid'][0])**2 +
                    (pred[1] - det['centroid'][1])**2
                )
                
                combined = min(dist, pred_dist)
                
                if combined < min_dist and combined < 150:
                    min_dist = combined
                    best_idx = i
            
            if best_idx >= 0:
                det = detections[best_idx]
                obj.update(det['bbox'], det['centroid'], det['mask'])
                used.add(best_idx)
        
        # Новые объекты
        for i, det in enumerate(detections):
            if i not in used and len(self.objects) < self.max_objects:
                self._create_object(det)
        
        self._cleanup_lost()
    
    def _create_object(self, det: Dict):
        """Создает новый объект"""
        obj = TrackedObject(
            id=self.next_id,
            color=self.colors[self.next_id % len(self.colors)]
        )
        obj.update(det['bbox'], det['centroid'], det['mask'])
        self.objects[self.next_id] = obj
        self.next_id += 1
    
    def _cleanup_lost(self, max_lost: int = 20):
        """Удаляет потерянные объекты"""
        to_remove = [oid for oid, obj in self.objects.items() 
                    if obj.lost_frames > max_lost]
        for oid in to_remove:
            del self.objects[oid]


@dataclass
class TrackedObject:
    """Отслеживаемый объект"""
    id: int
    color: Tuple[int, int, int]
    trajectory: deque = field(default_factory=lambda: deque(maxlen=80))
    bbox: Optional[Tuple[int, int, int, int]] = None
    centroid: Optional[Tuple[int, int]] = None
    mask: Optional[np.ndarray] = None
    lost_frames: int = 0
    
    def update(self, bbox, centroid, mask=None):
        self.bbox = bbox
        self.centroid = centroid
        self.mask = mask
        self.trajectory.append(centroid)
        self.lost_frames = 0
    
    def get_velocity(self) -> Tuple[float, float]:
        if len(self.trajectory) < 2:
            return (0, 0)
        pts = list(self.trajectory)[-5:]
        if len(pts) < 2:
            return (0, 0)
        return (
            (pts[-1][0] - pts[0][0]) / len(pts),
            (pts[-1][1] - pts[0][1]) / len(pts)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

class TheaterVisualizer:
    """Визуализатор для театрального трекинга"""
    
    def __init__(self, show_lighting_debug: bool = False):
        self.show_debug = show_lighting_debug
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def render(self, original: np.ndarray, filtered: np.ndarray,
               objects: List[TrackedObject], frame_idx: int,
               total_frames: int, fps: float) -> np.ndarray:
        """Рендерит финальный кадр"""
        
        # Используем отфильтрованный кадр как основу
        output = filtered.copy()
        
        # Рисуем маски
        for obj in objects:
            if obj.mask is not None:
                overlay = output.copy()
                overlay[obj.mask > 0] = obj.color
                cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)
                
                # Контур
                contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, contours, -1, obj.color, 2)
        
        # Рисуем траектории
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
        
        # Рисуем bboxes
        for obj in objects:
            if obj.bbox:
                x, y, w, h = obj.bbox
                
                # Уголки
                corner = min(25, w // 4, h // 4)
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
                label = f"Actor {obj.id}"
                cv2.putText(output, label, (x, y - 10),
                           self.font, 0.6, (0, 0, 0), 3)
                cv2.putText(output, label, (x, y - 10),
                           self.font, 0.6, obj.color, 1)
        
        # Центроиды
        for obj in objects:
            if obj.centroid:
                cv2.circle(output, obj.centroid, 6, (255, 255, 255), 2)
                cv2.circle(output, obj.centroid, 4, obj.color, -1)
        
        # Информация
        h, w = output.shape[:2]
        info_bg = output.copy()
        cv2.rectangle(info_bg, (10, 10), (280, 120), (0, 0, 0), -1)
        cv2.addWeighted(info_bg, 0.6, output, 0.4, 0, output)
        
        cv2.putText(output, "THEATER TRACKER", (20, 35),
                   self.font, 0.7, (100, 200, 255), 2)
        cv2.putText(output, f"Frame: {frame_idx + 1}/{total_frames}", (20, 60),
                   self.font, 0.5, (200, 200, 200), 1)
        cv2.putText(output, f"FPS: {fps:.1f}", (20, 80),
                   self.font, 0.5, (200, 200, 200), 1)
        cv2.putText(output, f"Objects: {len(objects)}", (20, 100),
                   self.font, 0.5, (200, 200, 200), 1)
        
        # Показываем debug если включено
        if self.show_debug:
            # Миниатюра оригинала
            mini_orig = cv2.resize(original, (160, 90))
            output[h-100:h-10, 10:170] = mini_orig
            cv2.putText(output, "Original", (15, h-105),
                       self.font, 0.4, (255, 255, 255), 1)
        
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# 🎬 ГЛАВНЫЙ ПРОЦЕССОР
# ═══════════════════════════════════════════════════════════════════════════════

class TheaterProcessor:
    """Главный процессор для театральных сцен"""
    
    def __init__(self, 
                 lighting_mode: str = 'adaptive',
                 mask_spotlights: bool = True,
                 show_debug: bool = False):
        
        self.tracker = TheaterObjectTracker(
            lighting_filter=lighting_mode,
            mask_spotlights=mask_spotlights
        )
        self.visualizer = TheaterVisualizer(show_lighting_debug=show_debug)
    
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
        
        print(f"\n{'═'*60}")
        print(f"  🎭 THEATER SCENE TRACKER")
        print(f"{'═'*60}")
        print(f"  📁 Input:  {input_path}")
        print(f"  📁 Output: {output_path}")
        print(f"  📐 Size:   {width}x{height}")
        print(f"  🎞  FPS:    {fps:.1f}")
        print(f"  📊 Frames: {total}")
        print(f"  💡 Lighting filter: {self.tracker.lighting_filter.mode}")
        print(f"  🔦 Mask spotlights: {self.tracker.mask_spotlights}")
        print(f"{'═'*60}")
        print(f"\n  Press 'Q' to quit, 'P' to pause\n")
        
        frame_times = []
        frame_idx = 0
        
        while frame_idx < total:
            ret, frame = cap.read()
            if not ret:
                break
            
            start = time.time()
            
            # Обработка
            objects, filtered = self.tracker.process_frame(frame)
            
            # Визуализация
            current_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else fps
            output_frame = self.visualizer.render(
                frame, filtered, objects, frame_idx, total, current_fps
            )
            
            frame_time = time.time() - start
            frame_times.append(frame_time)
            
            out.write(output_frame)
            
            if show_preview:
                preview = cv2.resize(output_frame, (0, 0), fx=0.6, fy=0.6)
                cv2.imshow('Theater Tracker', preview)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  Interrupted")
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
            
            if frame_idx % 30 == 0:
                progress = (frame_idx + 1) / total * 100
                avg_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
                print(f"\r  Processing: {progress:.1f}% | FPS: {avg_fps:.1f}", end="")
            
            frame_idx += 1
        
        print(f"\n\n{'═'*60}")
        print(f"  ✅ COMPLETE")
        print(f"{'═'*60}")
        if frame_times:
            print(f"  ⏱  Avg time: {np.mean(frame_times)*1000:.1f} ms")
            print(f"  🚀 Avg FPS:  {1.0/np.mean(frame_times):.1f}")
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
        description='🎭 Theater Scene Object Tracker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Lighting filter modes:
  light    - Minimal filtering (slight CLAHE)
  medium   - Moderate filtering (balanced)
  strong   - Heavy filtering (for very uneven lighting)
  adaptive - Auto-detects and applies best method (recommended)

Examples:
  python theater_tracker.py video.mp4
  python theater_tracker.py video.mp4 --lighting-filter strong
  python theater_tracker.py video.mp4 --no-mask-lights
  python theater_tracker.py video.mp4 --debug
        """
    )
    
    parser.add_argument('input', help='Input video path')
    parser.add_argument('-o', '--output', default=None, help='Output path')
    parser.add_argument('--lighting-filter', default='adaptive',
                       choices=['light', 'medium', 'strong', 'adaptive'],
                       help='Lighting filter mode')
    parser.add_argument('--no-mask-lights', action='store_true',
                       help='Disable spotlight masking')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug visualization')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable preview')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Max frames to process')
    
    args = parser.parse_args()
    
    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_theater.mp4")
    
    processor = TheaterProcessor(
        lighting_mode=args.lighting_filter,
        mask_spotlights=not args.no_mask_lights,
        show_debug=args.debug
    )
    
    processor.process_video(
        args.input,
        args.output,
        show_preview=not args.no_preview,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()


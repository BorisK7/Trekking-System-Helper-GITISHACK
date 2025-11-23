"""
Модуль для вычисления расстояний и размеров объектов
"""

import numpy as np


class DistanceEstimator:
    """Оценка расстояний и размеров объектов"""
    
    def __init__(self, camera_height_m=2.0, camera_angle_deg=0, focal_length_px=None):
        """
        Args:
            camera_height_m: Высота камеры над полом (метры)
            camera_angle_deg: Угол наклона камеры (градусы, 0 = горизонтально)
            focal_length_px: Фокусное расстояние в пикселях (если известно)
        """
        self.camera_height = camera_height_m
        self.camera_angle = np.radians(camera_angle_deg)
        self.focal_length = focal_length_px
        
        # Типичные высоты объектов (метры)
        self.typical_heights = {
            'person': 1.7,  # Средний человек
            'chair': 0.9,
            'couch': 0.8,
            'bed': 0.6,
            'dining table': 0.75,
            'car': 1.5,
            'truck': 2.5,
            'bus': 3.0
        }
    
    def estimate_distance_from_height(self, bbox, frame_height, class_name='person'):
        """
        Оценка расстояния на основе известной высоты объекта
        
        Args:
            bbox: (x1, y1, x2, y2) бокс объекта
            frame_height: Высота кадра в пикселях
            class_name: Тип объекта для получения типичной высоты
            
        Returns:
            Расстояние в метрах (приблизительное)
        """
        x1, y1, x2, y2 = bbox
        object_height_px = y2 - y1
        
        # Типичная высота объекта
        real_height = self.typical_heights.get(class_name, 1.7)
        
        # Простая модель: расстояние пропорционально отношению высот
        # Предполагаем, что объект на расстоянии 5м занимает примерно 1/4 кадра
        reference_distance = 5.0  # метры
        reference_height_ratio = 0.25
        
        current_height_ratio = object_height_px / frame_height
        
        if current_height_ratio > 0:
            distance = reference_distance * (reference_height_ratio / current_height_ratio)
            return distance
        
        return None
    
    def estimate_distance_from_focal_length(self, bbox, real_height_m, frame_height):
        """
        Оценка расстояния через фокусное расстояние (более точный метод)
        
        Args:
            bbox: (x1, y1, x2, y2)
            real_height_m: Реальная высота объекта в метрах
            frame_height: Высота кадра в пикселях
            
        Returns:
            Расстояние в метрах
        """
        if self.focal_length is None:
            return None
        
        x1, y1, x2, y2 = bbox
        object_height_px = y2 - y1
        
        if object_height_px > 0:
            # D = (f * H) / h
            # D - расстояние, f - фокусное расстояние, H - реальная высота, h - высота в пикселях
            distance = (self.focal_length * real_height_m) / object_height_px
            return distance
        
        return None
    
    def estimate_object_size(self, bbox, distance_m, frame_width, frame_height):
        """
        Оценка реального размера объекта
        
        Args:
            bbox: (x1, y1, x2, y2)
            distance_m: Расстояние до объекта в метрах
            frame_width, frame_height: Размеры кадра
            
        Returns:
            (width_m, height_m) размеры в метрах
        """
        if distance_m is None or distance_m <= 0:
            return None, None
        
        x1, y1, x2, y2 = bbox
        width_px = x2 - x1
        height_px = y2 - y1
        
        # Простая проекционная модель
        # Предполагаем FOV ~60 градусов по горизонтали
        fov_h = np.radians(60)
        fov_v = fov_h * (frame_height / frame_width)
        
        # Размер кадра на расстоянии distance_m
        frame_width_at_dist = 2 * distance_m * np.tan(fov_h / 2)
        frame_height_at_dist = 2 * distance_m * np.tan(fov_v / 2)
        
        # Размер объекта
        width_m = (width_px / frame_width) * frame_width_at_dist
        height_m = (height_px / frame_height) * frame_height_at_dist
        
        return width_m, height_m
    
    def distance_between_objects(self, bbox1, bbox2, distance1_m, distance2_m):
        """
        Оценка расстояния между двумя объектами
        
        Args:
            bbox1, bbox2: Боксы объектов (x1, y1, x2, y2)
            distance1_m, distance2_m: Расстояния объектов от камеры
            
        Returns:
            Расстояние между объектами в метрах
        """
        if distance1_m is None or distance2_m is None:
            return None
        
        # Центры объектов
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        
        # Расстояние в пикселях
        pixel_distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        
        # Средняя глубина
        avg_distance = (distance1_m + distance2_m) / 2
        
        # Преобразование пикселей в метры (упрощенно)
        # Предполагаем FOV ~60 градусов
        fov = np.radians(60)
        frame_diagonal = np.sqrt(1920**2 + 1080**2)  # Предполагаемое разрешение
        
        meters_per_pixel = (2 * avg_distance * np.tan(fov / 2)) / frame_diagonal
        real_distance = pixel_distance * meters_per_pixel
        
        # Учет глубины (теорема Пифагора в 3D)
        depth_diff = abs(distance1_m - distance2_m)
        total_distance = np.sqrt(real_distance**2 + depth_diff**2)
        
        return total_distance
    
    def calibrate_from_known_object(self, bbox, known_height_m, frame_height):
        """
        Калибровка фокусного расстояния по объекту известной высоты
        
        Args:
            bbox: (x1, y1, x2, y2) бокс известного объекта
            known_height_m: Известная высота в метрах
            frame_height: Высота кадра в пикселях
        """
        x1, y1, x2, y2 = bbox
        object_height_px = y2 - y1
        
        # Предполагаем расстояние (например, 3 метра)
        assumed_distance = 3.0
        
        # f = (h * D) / H
        self.focal_length = (object_height_px * assumed_distance) / known_height_m
    
    def check_social_distancing(self, person_bboxes, min_distance_m=2.0, frame_shape=(1080, 1920)):
        """
        Проверка соблюдения социальной дистанции
        
        Args:
            person_bboxes: [(x1, y1, x2, y2), ...] боксы людей
            min_distance_m: Минимальная дистанция в метрах
            frame_shape: (height, width) кадра
            
        Returns:
            List of (idx1, idx2, distance_m, is_violation) для каждой пары
        """
        violations = []
        frame_height, frame_width = frame_shape
        
        for i in range(len(person_bboxes)):
            bbox1 = person_bboxes[i]
            dist1 = self.estimate_distance_from_height(bbox1, frame_height, 'person')
            
            for j in range(i + 1, len(person_bboxes)):
                bbox2 = person_bboxes[j]
                dist2 = self.estimate_distance_from_height(bbox2, frame_height, 'person')
                
                if dist1 and dist2:
                    distance = self.distance_between_objects(bbox1, bbox2, dist1, dist2)
                    
                    if distance is not None:
                        is_violation = distance < min_distance_m
                        violations.append((i, j, distance, is_violation))
        
        return violations

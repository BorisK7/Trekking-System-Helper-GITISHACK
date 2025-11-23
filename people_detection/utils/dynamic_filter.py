"""
Модуль для фильтрации и классификации динамических объектов
"""

import numpy as np
from collections import deque
from .config import Config


class DynamicObjectFilter:
    """
    Фильтр для выделения только движущихся (динамических) объектов.
    Классифицирует объекты по типу движения.
    """
    
    def __init__(self, 
                 movement_threshold=5.0,      # Минимальное движение (пикселей)
                 history_length=10,           # Длина истории для анализа
                 min_frames_to_confirm=3):    # Минимум кадров для подтверждения
        """
        Args:
            movement_threshold: Порог движения в пикселях для считывания объекта динамическим
            history_length: Количество кадров для анализа движения
            min_frames_to_confirm: Минимум кадров с движением для подтверждения
        """
        self.movement_threshold = movement_threshold
        self.history_length = history_length
        self.min_frames_to_confirm = min_frames_to_confirm
        
        # История позиций для каждого трека
        self.track_history = {}  # {track_id: deque([(cx, cy, timestamp), ...])}
        
        # Классификация движения
        self.motion_classification = {}  # {track_id: 'walking', 'running', 'slow', etc.}
        
    def update(self, tracked_objects, frame_time):
        """
        Обновление фильтра и определение динамических объектов
        
        Args:
            tracked_objects: Список треков [(track_id, x1, y1, x2, y2, class_id, conf, state), ...]
            frame_time: Временная метка кадра
            
        Returns:
            dynamic_objects: Список только динамических объектов
            classifications: Словарь {track_id: classification_info}
        """
        from .common import unpack_track
        
        dynamic_objects = []
        classifications = {}
        
        current_track_ids = set()
        
        for track in tracked_objects:
            track_id, x1, y1, x2, y2, class_id, conf, state = unpack_track(track)
            current_track_ids.add(track_id)
            
            # Вычисляем центр
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Инициализация истории для нового трека
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.history_length)
            
            # Добавляем текущую позицию
            self.track_history[track_id].append((cx, cy, frame_time))
            
            # Анализ движения
            is_dynamic, motion_type, motion_metrics = self._analyze_motion(track_id, class_id)
            
            if is_dynamic:
                dynamic_objects.append(track)
                classifications[track_id] = {
                    'motion_type': motion_type,
                    'speed_px_s': motion_metrics['speed'],
                    'direction': motion_metrics['direction'],
                    'acceleration': motion_metrics['acceleration'],
                    'trajectory_length': motion_metrics['trajectory_length'],
                    'class_name': Config.COCO_CLASSES.get(class_id, 'unknown')
                }
        
        # Очистка истории для потерянных треков
        lost_tracks = set(self.track_history.keys()) - current_track_ids
        for track_id in lost_tracks:
            del self.track_history[track_id]
            if track_id in self.motion_classification:
                del self.motion_classification[track_id]
        
        return dynamic_objects, classifications
    
    def _analyze_motion(self, track_id, class_id):
        """
        Анализ движения объекта
        
        Returns:
            is_dynamic: bool - является ли объект динамическим
            motion_type: str - тип движения
            metrics: dict - метрики движения
        """
        history = self.track_history[track_id]
        
        # Недостаточно данных
        if len(history) < self.min_frames_to_confirm:
            return False, 'unknown', self._empty_metrics()
        
        # Вычисляем общее смещение
        positions = np.array([(x, y) for x, y, _ in history])
        
        # Общее пройденное расстояние
        total_distance = 0
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i-1])
            total_distance += dist
        
        # Средняя скорость
        time_span = history[-1][2] - history[0][2]
        avg_speed = total_distance / time_span if time_span > 0 else 0
        
        # Проверка на динамичность
        if total_distance < self.movement_threshold:
            return False, 'static', self._empty_metrics()
        
        # Вектор направления (от начала к концу)
        direction_vector = positions[-1] - positions[0]
        direction_angle = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))
        
        # Ускорение (изменение скорости)
        acceleration = self._calculate_acceleration(history)
        
        # Классификация типа движения
        motion_type = self._classify_motion_type(avg_speed, acceleration, class_id, total_distance)
        
        metrics = {
            'speed': avg_speed,
            'direction': direction_angle,
            'acceleration': acceleration,
            'trajectory_length': total_distance
        }
        
        return True, motion_type, metrics
    
    def _calculate_acceleration(self, history):
        """Расчет ускорения (изменение скорости)"""
        if len(history) < 3:
            return 0.0
        
        speeds = []
        for i in range(1, len(history)):
            pos1 = np.array(history[i-1][:2])
            pos2 = np.array(history[i][:2])
            dt = history[i][2] - history[i-1][2]
            
            if dt > 0:
                speed = np.linalg.norm(pos2 - pos1) / dt
                speeds.append(speed)
        
        if len(speeds) < 2:
            return 0.0
        
        # Изменение скорости
        speed_changes = [speeds[i] - speeds[i-1] for i in range(1, len(speeds))]
        return np.mean(speed_changes) if speed_changes else 0.0
    
    def _classify_motion_type(self, speed, acceleration, class_id, distance):
        """
        Классификация типа движения
        
        Returns:
            motion_type: 'slow_walk', 'walk', 'fast_walk', 'run', 'sprint', 
                        'vehicle_slow', 'vehicle_fast', 'erratic'
        """
        class_name = Config.COCO_CLASSES.get(class_id, 'unknown')
        
        # Для людей
        if class_name == 'person':
            if speed < 20:
                return 'slow_walk'
            elif speed < 50:
                return 'walk'
            elif speed < 100:
                return 'fast_walk'
            elif speed < 200:
                return 'run'
            else:
                return 'sprint'
        
        # Для транспорта
        elif class_id in [1, 2, 3, 5, 6, 7, 8]:  # bicycle, car, motorcycle, bus, etc.
            if speed < 50:
                return 'vehicle_slow'
            elif speed < 150:
                return 'vehicle_medium'
            else:
                return 'vehicle_fast'
        
        # Для животных
        elif class_id in [14, 15, 16, 17, 18, 19]:  # bird, cat, dog, etc.
            if abs(acceleration) > 10:
                return 'animal_erratic'
            elif speed < 30:
                return 'animal_slow'
            else:
                return 'animal_fast'
        
        # Для остальных объектов
        else:
            if abs(acceleration) > 5:
                return 'erratic'
            elif speed < 30:
                return 'slow_motion'
            else:
                return 'fast_motion'
    
    def _empty_metrics(self):
        """Пустые метрики для статических объектов"""
        return {
            'speed': 0.0,
            'direction': 0.0,
            'acceleration': 0.0,
            'trajectory_length': 0.0
        }
    
    def get_statistics(self):
        """Получение статистики по типам движения"""
        stats = {
            'total_tracked': len(self.track_history),
            'motion_types': {}
        }
        
        for track_id, motion_type in self.motion_classification.items():
            stats['motion_types'][motion_type] = stats['motion_types'].get(motion_type, 0) + 1
        
        return stats


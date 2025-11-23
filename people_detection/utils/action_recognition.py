import numpy as np
from collections import deque
from .geometry import calculate_angle

class ActionRecognizer:
    """
    Класс для распознавания действий на основе поз (скелета) и движения.
    Использует ключевые точки COCO (17 точек).
    """
    
    # Индексы ключевых точек COCO
    # 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
    # 5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow, 9: L-Wrist, 10: R-Wrist
    # 11: L-Hip, 12: R-Hip, 13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle
    
    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        # История поз и скоростей для каждого трека
        # {track_id: {'keypoints': deque, 'velocity': deque, 'last_action': str}}
        self.history = {}
        
        # Пороги для действий
        self.velocity_threshold_walk = 1.5  # пикселей/кадр (зависит от разрешения)
        self.velocity_threshold_run = 6.0
        self.fall_aspect_ratio = 1.5  # ширина / высота
        self.sit_knee_angle = 110  # градусов

    def update(self, track_id, keypoints, bbox, velocity_vector=None):
        """
        Обновление состояния для трека и определение действия
        
        Args:
            track_id: ID трека
            keypoints: numpy array (17, 3) [x, y, conf]
            bbox: [x1, y1, x2, y2]
            velocity_vector: (vx, vy) скорость центра
            
        Returns:
            action_label: строка с действием
            confidence: уверенность (0.0-1.0)
        """
        if track_id not in self.history:
            self.history[track_id] = {
                'keypoints': deque(maxlen=self.buffer_size),
                'bbox_history': deque(maxlen=self.buffer_size),
                'last_action': 'standing',
                'consecutive_falls': 0
            }
            
        state = self.history[track_id]
        state['keypoints'].append(keypoints)
        state['bbox_history'].append(bbox)
        
        # Очистка старых треков (опционально, здесь простая логика)
        if len(self.history) > 100:
            # Удаляем самые старые ключи, если буфер переполнен
            pass

        return self._classify_action(keypoints, bbox, velocity_vector, state)

    def _classify_action(self, kpts, bbox, velocity, state):
        """Логика классификации"""
        # Проверка на корректность ключевых точек
        if kpts is None or len(kpts) == 0:
            return "unknown", 0.0

        # 1. FALLING (Падение)
        # Логика: резкое изменение соотношения сторон (становится широким) + центр смещается вниз
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        aspect_ratio = w / h if h > 0 else 0
        
        if aspect_ratio > self.fall_aspect_ratio:
            state['consecutive_falls'] += 1
        else:
            state['consecutive_falls'] = max(0, state['consecutive_falls'] - 1)
            
        if state['consecutive_falls'] >= 5: # 5 кадров подряд лежим
            return "falling", 0.85

        # 2. SITTING (Сидение)
        # Логика: угол в коленях ~90 градусов или бедра близко к коленям по высоте
        l_hip, r_hip = kpts[11], kpts[12]
        l_knee, r_knee = kpts[13], kpts[14]
        l_ankle, r_ankle = kpts[15], kpts[16]
        
        # Проверяем уверенность точек ног
        if l_hip[2] > 0.5 and l_knee[2] > 0.5 and l_ankle[2] > 0.5:
            angle_l = calculate_angle(l_hip, l_knee, l_ankle)
            if angle_l < self.sit_knee_angle:
                return "sitting", 0.8
        
        if r_hip[2] > 0.5 and r_knee[2] > 0.5 and r_ankle[2] > 0.5:
            angle_r = calculate_angle(r_hip, r_knee, r_ankle)
            if angle_r < self.sit_knee_angle:
                return "sitting", 0.8

        # 3. MOVEMENT (Движение)
        speed = 0
        if velocity:
            vx, vy = velocity
            speed = (vx**2 + vy**2)**0.5
        
        if speed > self.velocity_threshold_run:
            return "running", 0.7
        elif speed > self.velocity_threshold_walk:
            return "walking", 0.6

        # 4. STANDING (Стояние - по умолчанию)
        return "standing", 0.5
    
    def cleanup(self, active_track_ids):
        """Удаление данных потерянных треков"""
        current_ids = set(self.history.keys())
        active_ids = set(active_track_ids)
        to_remove = current_ids - active_ids
        for tid in to_remove:
            del self.history[tid]


"""
Модуль для трекинга объектов, вычисления траекторий и скорости
"""

import numpy as np
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment


class ObjectTracker:
    """Трекер объектов с расчетом траекторий и скорости"""
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, max_trajectory_length=100):
        """
        Args:
            max_age: Максимальное количество кадров без детекции перед удалением трека
            min_hits: Минимальное количество детекций для активации трека
            iou_threshold: Порог IoU для ассоциации детекций с треками
            max_trajectory_length: Максимальная длина траектории для визуализации
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_trajectory_length = max_trajectory_length
        
        # Треки
        self.tracks = {}
        self.next_id = 1
        
        # Статистика
        self.total_tracks_created = 0
        self.active_tracks_count = 0
        
    def update(self, detections, frame_time):
        """
        Обновление треков на основе новых детекций
        
        Args:
            detections: [(x1, y1, x2, y2, class_id, conf), ...]
            frame_time: Временная метка кадра (сек)
            
        Returns:
            List of tracked objects with IDs: [(track_id, x1, y1, x2, y2, class_id, conf), ...]
        """
        # Если детекций нет
        if len(detections) == 0:
            self._age_tracks()
            return self._get_active_tracks()
        
        # Извлечение боксов
        det_boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
        det_classes = np.array([d[4] for d in detections])
        det_confs = np.array([d[5] for d in detections])
        
        # Предсказание позиций существующих треков
        track_ids = list(self.tracks.keys())
        if len(track_ids) > 0:
            track_boxes = np.array([self.tracks[tid]['bbox'] for tid in track_ids])
            
            # Вычисление IoU матрицы
            iou_matrix = self._compute_iou_matrix(det_boxes, track_boxes)
            
            # Фильтрация по классам (треки должны совпадать с классом детекции)
            for i, det_class in enumerate(det_classes):
                for j, tid in enumerate(track_ids):
                    if self.tracks[tid]['class_id'] != det_class:
                        iou_matrix[i, j] = 0
            
            # Ассоциация детекций с треками (Hungarian algorithm)
            matched_indices = self._associate(iou_matrix)
            
            matched_dets = set()
            matched_tracks = set()
            
            # Обновление сопоставленных треков
            for det_idx, track_idx in matched_indices:
                if iou_matrix[det_idx, track_idx] >= self.iou_threshold:
                    tid = track_ids[track_idx]
                    self._update_track(tid, det_boxes[det_idx], det_confs[det_idx], frame_time)
                    matched_dets.add(det_idx)
                    matched_tracks.add(track_idx)
            
            # Старение несопоставленных треков
            for j, tid in enumerate(track_ids):
                if j not in matched_tracks:
                    self.tracks[tid]['age'] += 1
                    self.tracks[tid]['hits_streak'] = 0
        else:
            matched_dets = set()
        
        # Создание новых треков для несопоставленных детекций
        for i in range(len(detections)):
            if i not in matched_dets:
                self._create_track(det_boxes[i], det_classes[i], det_confs[i], frame_time)
        
        # Удаление старых треков
        self._remove_old_tracks()
        
        return self._get_active_tracks()
    
    def _create_track(self, bbox, class_id, conf, frame_time):
        """Создание нового трека"""
        track_id = self.next_id
        self.next_id += 1
        self.total_tracks_created += 1
        
        cx, cy = self._get_center(bbox)
        
        self.tracks[track_id] = {
            'bbox': bbox,
            'class_id': class_id,
            'conf': conf,
            'age': 0,
            'hits': 1,
            'hits_streak': 1,
            'trajectory': deque([(cx, cy, frame_time)], maxlen=self.max_trajectory_length),
            'velocity': (0, 0),  # пиксели/сек
            'first_seen': frame_time,
            'last_seen': frame_time,
            'total_distance': 0.0  # пиксели
        }
    
    def _update_track(self, track_id, bbox, conf, frame_time):
        """Обновление существующего трека"""
        track = self.tracks[track_id]
        
        # Вычисление скорости
        prev_cx, prev_cy, prev_time = track['trajectory'][-1]
        cx, cy = self._get_center(bbox)
        dt = frame_time - prev_time
        
        if dt > 0:
            vx = (cx - prev_cx) / dt
            vy = (cy - prev_cy) / dt
            
            # Сглаживание скорости (экспоненциальное скользящее среднее)
            alpha = 0.3
            track['velocity'] = (
                alpha * vx + (1 - alpha) * track['velocity'][0],
                alpha * vy + (1 - alpha) * track['velocity'][1]
            )
            
            # Общее пройденное расстояние
            distance = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            track['total_distance'] += distance
        
        # Обновление данных трека
        track['bbox'] = bbox
        track['conf'] = conf
        track['age'] = 0
        track['hits'] += 1
        track['hits_streak'] += 1
        track['last_seen'] = frame_time
        track['trajectory'].append((cx, cy, frame_time))
    
    def _age_tracks(self):
        """Старение всех треков (когда нет детекций)"""
        for tid in list(self.tracks.keys()):
            self.tracks[tid]['age'] += 1
            self.tracks[tid]['hits_streak'] = 0
    
    def _remove_old_tracks(self):
        """Удаление старых треков"""
        to_remove = []
        for tid, track in self.tracks.items():
            if track['age'] > self.max_age:
                to_remove.append(tid)
        
        for tid in to_remove:
            del self.tracks[tid]
    
    def _get_active_tracks(self):
        """Получение активных треков (с достаточным количеством детекций)"""
        active = []
        for tid, track in self.tracks.items():
            if track['hits'] >= self.min_hits or track['hits_streak'] >= self.min_hits:
                x1, y1, x2, y2 = track['bbox']
                active.append((tid, x1, y1, x2, y2, track['class_id'], track['conf']))
        
        self.active_tracks_count = len(active)
        return active
    
    def _compute_iou_matrix(self, boxes1, boxes2):
        """Вычисление IoU матрицы между двумя наборами боксов"""
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))
        
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = self._compute_iou(box1, box2)
        
        return iou_matrix
    
    def _compute_iou(self, box1, box2):
        """Вычисление IoU между двумя боксами"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Пересечение
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Объединение
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def _associate(self, iou_matrix):
        """Ассоциация детекций с треками через Hungarian algorithm"""
        if iou_matrix.size == 0:
            return []
        
        # Преобразование IoU в стоимость (1 - IoU)
        cost_matrix = 1 - iou_matrix
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        return list(zip(row_ind, col_ind))
    
    def _get_center(self, bbox):
        """Получение центра бокса"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2
    
    def get_trajectory(self, track_id, max_points=None):
        """Получение траектории трека"""
        if track_id not in self.tracks:
            return []
        
        trajectory = list(self.tracks[track_id]['trajectory'])
        
        if max_points and len(trajectory) > max_points:
            trajectory = trajectory[-max_points:]
        
        return [(x, y) for x, y, _ in trajectory]
    
    def get_velocity(self, track_id):
        """Получение скорости трека (пиксели/сек)"""
        if track_id not in self.tracks:
            return (0, 0)
        return self.tracks[track_id]['velocity']
    
    def get_speed(self, track_id):
        """Получение модуля скорости (пиксели/сек)"""
        vx, vy = self.get_velocity(track_id)
        return np.sqrt(vx**2 + vy**2)
    
    def get_track_duration(self, track_id):
        """Получение длительности существования трека (секунды)"""
        if track_id not in self.tracks:
            return 0
        track = self.tracks[track_id]
        return track['last_seen'] - track['first_seen']
    
    def get_total_distance(self, track_id):
        """Получение общего пройденного расстояния (пиксели)"""
        if track_id not in self.tracks:
            return 0
        return self.tracks[track_id]['total_distance']
    
    def get_statistics(self):
        """Получение общей статистики трекинга"""
        return {
            'total_tracks_created': self.total_tracks_created,
            'active_tracks': self.active_tracks_count,
            'total_tracks': len(self.tracks),
            'tracks_by_class': self._count_by_class()
        }
    
    def _count_by_class(self):
        """Подсчет треков по классам"""
        from .config import Config
        counts = defaultdict(int)
        for track in self.tracks.values():
            if track['hits'] >= self.min_hits:
                class_name = Config.COCO_CLASSES.get(track['class_id'], 'unknown')
                counts[class_name] += 1
        return dict(counts)

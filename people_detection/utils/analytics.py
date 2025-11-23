"""
Модуль для расширенной аналитики: heat map, зоны, занятость, статистика
"""

import numpy as np
import cv2
from collections import defaultdict, deque
from .config import Config
from .geometry import point_in_polygon, point_side_of_line, boxes_overlap_iou
from .common import unpack_track


class SceneAnalytics:
    """Аналитика сцены: heat map, зоны, плотность"""
    
    def __init__(self, frame_shape, heat_decay=0.95):
        """
        Args:
            frame_shape: (height, width) размер кадра
            heat_decay: Коэффициент затухания heat map (0-1)
        """
        self.height, self.width = frame_shape[:2]
        self.heat_decay = heat_decay
        
        # Heat map (тепловая карта активности)
        self.heat_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Зоны интереса
        self.zones = {}  # {zone_id: {'polygon': [(x,y),...], 'count': 0, 'class_counts': {}}}
        
        # Статистика по времени
        self.timeline = {
            'timestamps': [],
            'total_objects': [],
            'objects_by_class': defaultdict(list)
        }
        
        # Линии подсчета (для входа/выхода)
        self.counting_lines = {}  # {line_id: {'start': (x,y), 'end': (x,y), 'crossed': set()}}
        
    def update_heat_map(self, tracked_objects):
        """
        Обновление heat map на основе позиций объектов
        """
        # Затухание предыдущего heat map
        self.heat_map *= self.heat_decay
        
        # Добавление текущих позиций
        for track in tracked_objects:
            track_id, x1, y1, x2, y2, class_id, conf, _ = unpack_track(track)
            
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Добавление гауссового пятна вокруг центра
            self._add_gaussian_spot(cx, cy, radius=30, intensity=10)
    
    def _add_gaussian_spot(self, cx, cy, radius=30, intensity=10):
        """Добавление гауссового пятна на heat map"""
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        gaussian = np.exp(-(x**2 + y**2) / (2 * (radius/2)**2)) * intensity
        
        # Границы на heat map
        y1 = max(0, cy - radius)
        y2 = min(self.height, cy + radius + 1)
        x1 = max(0, cx - radius)
        x2 = min(self.width, cx + radius + 1)
        
        # Границы на gaussian
        gy1 = max(0, radius - cy)
        gy2 = gy1 + (y2 - y1)
        gx1 = max(0, radius - cx)
        gx2 = gx1 + (x2 - x1)
        
        # Применение
        if y2 > y1 and x2 > x1:
            self.heat_map[y1:y2, x1:x2] += gaussian[gy1:gy2, gx1:gx2]
    
    def get_heat_map_overlay(self, alpha=0.5):
        """
        Получение heat map как цветного overlay для визуализации
        """
        # Нормализация
        normalized = self.heat_map / (self.heat_map.max() + 1e-6)
        normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        
        # Применение цветовой карты
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return colored, alpha
    
    def define_zone(self, zone_id, polygon):
        """
        Определение зоны интереса
        """
        self.zones[zone_id] = {
            'polygon': polygon,
            'count': 0,
            'class_counts': defaultdict(int),
            'track_ids': set()
        }
    
    def update_zones(self, tracked_objects):
        """
        Обновление статистики по зонам
        """
        
        # Сброс счетчиков
        for zone in self.zones.values():
            zone['count'] = 0
            zone['class_counts'].clear()
            zone['track_ids'].clear()
        
        # Проверка объектов в зонах
        for track in tracked_objects:
            track_id, x1, y1, x2, y2, class_id, conf, _ = unpack_track(track)
            
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            for zone_id, zone in self.zones.items():
                if point_in_polygon((cx, cy), zone['polygon']):
                    zone['count'] += 1
                    class_name = Config.COCO_CLASSES.get(class_id, 'unknown')
                    zone['class_counts'][class_name] += 1
                    zone['track_ids'].add(track_id)
    
    def define_counting_line(self, line_id, start, end):
        """
        Определение линии подсчета для входа/выхода
        """
        self.counting_lines[line_id] = {
            'start': start,
            'end': end,
            'crossed': set(),  # track_ids которые пересекли
            'cross_count': 0,
            'prev_positions': {}  # {track_id: 'above'/'below'}
        }
    
    def update_counting_lines(self, tracked_objects):
        """
        Обновление подсчета пересечений линий
        """
        for track in tracked_objects:
            track_id, x1, y1, x2, y2, class_id, conf, _ = unpack_track(track)
            
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            for line_id, line in self.counting_lines.items():
                # Определение положения относительно линии
                position = point_side_of_line((cx, cy), line['start'], line['end'])
                
                # Проверка пересечения
                if track_id in line['prev_positions']:
                    prev_pos = line['prev_positions'][track_id]
                    if prev_pos != position and track_id not in line['crossed']:
                        # Пересечение!
                        line['crossed'].add(track_id)
                        line['cross_count'] += 1
                
                line['prev_positions'][track_id] = position
    
    def update_timeline(self, timestamp, tracked_objects):
        """
        Обновление временной статистики
        """
        
        self.timeline['timestamps'].append(timestamp)
        self.timeline['total_objects'].append(len(tracked_objects))
        
        # Подсчет по классам
        class_counts = defaultdict(int)
        for track in tracked_objects:
            _, _, _, _, _, class_id, _, _ = unpack_track(track)
            class_name = Config.COCO_CLASSES.get(class_id, 'unknown')
            class_counts[class_name] += 1
        
        for class_name, count in class_counts.items():
            self.timeline['objects_by_class'][class_name].append(count)
    
    def get_zone_statistics(self):
        """Получение статистики по зонам"""
        stats = {}
        for zone_id, zone in self.zones.items():
            stats[zone_id] = {
                'total_count': zone['count'],
                'by_class': dict(zone['class_counts']),
                'unique_tracks': len(zone['track_ids'])
            }
        return stats
    
    def get_line_statistics(self):
        """Получение статистики по линиям подсчета"""
        stats = {}
        for line_id, line in self.counting_lines.items():
            stats[line_id] = {
                'cross_count': line['cross_count'],
                'unique_crossers': len(line['crossed'])
            }
        return stats


class OccupancyAnalyzer:
    """Анализатор занятости мест (стульев, парковочных мест и т.д.)"""
    
    def __init__(self):
        """Инициализация анализатора занятости"""
        self.furniture_positions = {}  # {object_id: {'bbox': (x1,y1,x2,y2), 'class': 'chair'}}
        self.occupancy_status = {}  # {object_id: True/False}
        self.occupancy_history = defaultdict(list)  # {object_id: [True, False, ...]}
        
    def register_furniture(self, object_id, bbox, class_name):
        """
        Регистрация мебели для отслеживания занятости
        """
        self.furniture_positions[object_id] = {
            'bbox': bbox,
            'class': class_name
        }
        self.occupancy_status[object_id] = False
    
    def update_occupancy(self, people_detections, furniture_detections):
        """
        Обновление статуса занятости
        
        Args:
            people_detections: [(track_id, x1, y1, x2, y2, class_id, conf), ...] людей
            furniture_detections: [(track_id, x1, y1, x2, y2, class_id, conf), ...] мебели
        """
        # Сброс статуса
        for obj_id in self.occupancy_status:
            self.occupancy_status[obj_id] = False
        
        # Проверка пересечений людей с мебелью
        for person_track in people_detections:
            # Используем unpack_track для стандартизации
            _, px1, py1, px2, py2, _, _, _ = unpack_track(person_track)
            person_bbox = (px1, py1, px2, py2)
            
            for furn_track in furniture_detections:
                # Аналогично для мебели
                furn_id, fx1, fy1, fx2, fy2, _, _, _ = unpack_track(furn_track)
                furn_bbox = (fx1, fy1, fx2, fy2)
                
                # Проверка пересечения
                if boxes_overlap_iou(person_bbox, furn_bbox) >= 0.3:
                    # Мебель занята
                    if furn_id in self.occupancy_status:
                        self.occupancy_status[furn_id] = True
        
        # Сохранение истории
        for obj_id, status in self.occupancy_status.items():
            self.occupancy_history[obj_id].append(status)
    
    def get_occupancy_rate(self):
        """Получение процента занятости"""
        if not self.occupancy_status:
            return 0.0
        
        occupied = sum(1 for status in self.occupancy_status.values() if status)
        total = len(self.occupancy_status)
        
        return (occupied / total) * 100.0
    
    def get_free_spots(self):
        """Получение списка свободных мест"""
        return [obj_id for obj_id, status in self.occupancy_status.items() if not status]
    
    def get_statistics(self):
        """Получение статистики занятости"""
        total = len(self.occupancy_status)
        occupied = sum(1 for status in self.occupancy_status.values() if status)
        free = total - occupied
        
        return {
            'total_spots': total,
            'occupied': occupied,
            'free': free,
            'occupancy_rate': self.get_occupancy_rate()
        }


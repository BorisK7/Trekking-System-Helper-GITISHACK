"""
Модуль для визуализации траекторий, heat map, зон и статистики
"""

import cv2
import numpy as np
from .config import Config


class TrajectoryVisualizer:
    """Визуализация траекторий движения объектов"""
    
    def __init__(self):
        """Инициализация визуализатора"""
        self.trajectory_colors = {}  # {track_id: (B,G,R)}
        self.color_palette = self._generate_color_palette(100)
        
    def _generate_color_palette(self, n):
        """Генерация палитры различимых цветов"""
        colors = []
        for i in range(n):
            hue = int((i * 137.5) % 180)  # Золотое сечение для разнообразия
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color_bgr)))
        return colors
    
    def get_color_for_track(self, track_id):
        """Получение цвета для трека"""
        if track_id not in self.trajectory_colors:
            color_idx = track_id % len(self.color_palette)
            self.trajectory_colors[track_id] = self.color_palette[color_idx]
        return self.trajectory_colors[track_id]
    
    def draw_trajectories(self, frame, tracker, min_length=5, thickness=4):
        """
        Отрисовка траекторий движения
        
        Args:
            frame: Кадр для отрисовки
            tracker: ObjectTracker с траекториями
            min_length: Минимальная длина траектории для отрисовки
            thickness: Толщина линии (по умолчанию 4)
        """
        for track_id, track in tracker.tracks.items():
            trajectory = track['trajectory']
            
            if len(trajectory) < min_length:
                continue
            
            # Цвет траектории
            color = self.get_color_for_track(track_id)
            
            # Отрисовка линии траектории
            points = [(int(x), int(y)) for x, y, _ in trajectory]
            
            # Градиент прозрачности (старые точки более прозрачные)
            for i in range(1, len(points)):
                # Альфа от 0.3 до 1.0
                alpha = 0.3 + (i / len(points)) * 0.7
                thickness_current = max(1, int(thickness * alpha))
                
                cv2.line(frame, points[i-1], points[i], color, thickness_current)
            
            # Стрелка направления на последнем сегменте
            if len(points) >= 2:
                self._draw_arrow(frame, points[-2], points[-1], color, thickness)
        
        return frame
    
    def _draw_arrow(self, frame, pt1, pt2, color, thickness):
        """Отрисовка стрелки направления"""
        # Вектор направления
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 10:  # Слишком короткий сегмент
            return
        
        # Нормализация
        dx /= length
        dy /= length
        
        # Размер стрелки
        arrow_len = min(15, length * 0.3)
        arrow_angle = np.pi / 6  # 30 градусов
        
        # Точки стрелки
        arrow_pt1 = (
            int(pt2[0] - arrow_len * (dx * np.cos(arrow_angle) + dy * np.sin(arrow_angle))),
            int(pt2[1] - arrow_len * (dy * np.cos(arrow_angle) - dx * np.sin(arrow_angle)))
        )
        arrow_pt2 = (
            int(pt2[0] - arrow_len * (dx * np.cos(arrow_angle) - dy * np.sin(arrow_angle))),
            int(pt2[1] - arrow_len * (dy * np.cos(arrow_angle) + dx * np.sin(arrow_angle)))
        )
        
        cv2.line(frame, pt2, arrow_pt1, color, thickness)
        cv2.line(frame, pt2, arrow_pt2, color, thickness)
    
    def draw_track_info(self, frame, tracked_objects, tracker, show_speed=True, show_id=True):
        """
        Отрисовка информации о треках
        
        Args:
            frame: Кадр
            tracked_objects: [(track_id, x1, y1, x2, y2, class_id, conf), ...]
            tracker: ObjectTracker
            show_speed: Показывать скорость
            show_id: Показывать ID трека
        """
        for track_id, x1, y1, x2, y2, class_id, conf in tracked_objects:
            color = self.get_color_for_track(track_id)
            class_name = Config.COCO_CLASSES.get(class_id, 'unknown')
            
            # Информация
            info_lines = []
            if show_id:
                info_lines.append(f"ID:{track_id}")
            info_lines.append(f"{class_name.title()}")
            
            if show_speed:
                speed_px_s = tracker.get_speed(track_id)
                info_lines.append(f"{speed_px_s:.1f}px/s")
            
            # Отрисовка текста
            y_offset = int(y1) - 10
            for line in reversed(info_lines):
                (tw, th), _ = cv2.getTextSize(line, Config.FONT, 0.5, 1)
                cv2.rectangle(
                    frame,
                    (int(x1), y_offset - th - 2),
                    (int(x1) + tw, y_offset),
                    color,
                    -1
                )
                cv2.putText(
                    frame,
                    line,
                    (int(x1), y_offset - 2),
                    Config.FONT,
                    0.5,
                    (255, 255, 255),
                    1
                )
                y_offset -= (th + 4)
        
        return frame


class HeatMapVisualizer:
    """Визуализация heat map активности"""
    
    @staticmethod
    def overlay_heat_map(frame, heat_map, alpha=0.5):
        """
        Наложение heat map на кадр
        
        Args:
            frame: Исходный кадр
            heat_map: Heat map изображение
            alpha: Прозрачность heat map
        """
        # Изменение размера heat map под кадр
        if heat_map.shape[:2] != frame.shape[:2]:
            heat_map = cv2.resize(heat_map, (frame.shape[1], frame.shape[0]))
        
        # Наложение
        result = cv2.addWeighted(frame, 1 - alpha, heat_map, alpha, 0)
        return result


class ZoneVisualizer:
    """Визуализация зон интереса и линий подсчета"""
    
    @staticmethod
    def draw_zones(frame, zones, analytics, show_stats=True):
        """
        Отрисовка зон интереса
        
        Args:
            frame: Кадр
            zones: {zone_id: {'polygon': [(x,y),...], ...}}
            analytics: SceneAnalytics с обновленной статистикой
            show_stats: Показывать статистику по зонам
        """
        for zone_id, zone in zones.items():
            polygon = np.array(zone['polygon'], dtype=np.int32)
            
            # Цвет в зависимости от количества объектов
            count = zone.get('count', 0)
            if count == 0:
                color = (100, 100, 100)  # Серый - пусто
            elif count < 3:
                color = (0, 255, 0)  # Зеленый - мало
            elif count < 6:
                color = (0, 165, 255)  # Оранжевый - средне
            else:
                color = (0, 0, 255)  # Красный - много
            
            # Полупрозрачная заливка
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Контур
            cv2.polylines(frame, [polygon], True, color, 2)
            
            # Статистика
            if show_stats and count > 0:
                # Центр зоны
                M = cv2.moments(polygon)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    text = f"Zone {zone_id}: {count}"
                    cv2.putText(
                        frame,
                        text,
                        (cx - 40, cy),
                        Config.FONT,
                        0.6,
                        (255, 255, 255),
                        2
                    )
        
        return frame
    
    @staticmethod
    def draw_counting_lines(frame, counting_lines, show_count=True):
        """
        Отрисовка линий подсчета
        
        Args:
            frame: Кадр
            counting_lines: {line_id: {'start': (x,y), 'end': (x,y), ...}}
            show_count: Показывать счетчик
        """
        for line_id, line in counting_lines.items():
            start = tuple(map(int, line['start']))
            end = tuple(map(int, line['end']))
            count = line.get('cross_count', 0)
            
            # Линия
            cv2.line(frame, start, end, (255, 0, 255), 3)
            
            # Счетчик
            if show_count:
                mid_x = (start[0] + end[0]) // 2
                mid_y = (start[1] + end[1]) // 2
                
                text = f"Line {line_id}: {count}"
                (tw, th), _ = cv2.getTextSize(text, Config.FONT, 0.7, 2)
                
                cv2.rectangle(
                    frame,
                    (mid_x - tw//2 - 5, mid_y - th - 5),
                    (mid_x + tw//2 + 5, mid_y + 5),
                    (255, 0, 255),
                    -1
                )
                cv2.putText(
                    frame,
                    text,
                    (mid_x - tw//2, mid_y),
                    Config.FONT,
                    0.7,
                    (255, 255, 255),
                    2
                )
        
        return frame


class StatisticsOverlay:
    """Наложение статистики на кадр"""
    
    @staticmethod
    def draw_statistics_panel(frame, tracker, analytics=None, occupancy=None):
        """
        Отрисовка панели со статистикой
        
        Args:
            frame: Кадр
            tracker: ObjectTracker
            analytics: SceneAnalytics (опционально)
            occupancy: OccupancyAnalyzer (опционально)
        """
        h, w = frame.shape[:2]
        panel_width = 300
        panel_height = h
        
        # Создание панели
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        # Статистика трекинга
        y = 30
        cv2.putText(panel, "TRACKING STATISTICS", (10, y), Config.FONT, 0.6, (255, 255, 255), 2)
        y += 30
        
        stats = tracker.get_statistics()
        cv2.putText(panel, f"Active tracks: {stats['active_tracks']}", (10, y), Config.FONT, 0.5, (200, 200, 200), 1)
        y += 25
        cv2.putText(panel, f"Total created: {stats['total_tracks_created']}", (10, y), Config.FONT, 0.5, (200, 200, 200), 1)
        y += 30
        
        # По классам
        if stats['tracks_by_class']:
            cv2.putText(panel, "By class:", (10, y), Config.FONT, 0.5, (255, 200, 100), 1)
            y += 25
            for class_name, count in sorted(stats['tracks_by_class'].items(), key=lambda x: -x[1])[:5]:
                cv2.putText(panel, f"  {class_name}: {count}", (10, y), Config.FONT, 0.45, (200, 200, 200), 1)
                y += 20
        
        # Статистика зон
        if analytics:
            y += 20
            cv2.putText(panel, "ZONE STATISTICS", (10, y), Config.FONT, 0.6, (255, 255, 255), 2)
            y += 30
            
            zone_stats = analytics.get_zone_statistics()
            for zone_id, zstats in zone_stats.items():
                if zstats['total_count'] > 0:
                    cv2.putText(panel, f"Zone {zone_id}: {zstats['total_count']}", (10, y), Config.FONT, 0.5, (200, 200, 200), 1)
                    y += 20
        
        # Занятость
        if occupancy:
            y += 20
            cv2.putText(panel, "OCCUPANCY", (10, y), Config.FONT, 0.6, (255, 255, 255), 2)
            y += 30
            
            occ_stats = occupancy.get_statistics()
            cv2.putText(panel, f"Total spots: {occ_stats['total_spots']}", (10, y), Config.FONT, 0.5, (200, 200, 200), 1)
            y += 25
            cv2.putText(panel, f"Occupied: {occ_stats['occupied']}", (10, y), Config.FONT, 0.5, (0, 0, 255), 1)
            y += 25
            cv2.putText(panel, f"Free: {occ_stats['free']}", (10, y), Config.FONT, 0.5, (0, 255, 0), 1)
            y += 25
            cv2.putText(panel, f"Rate: {occ_stats['occupancy_rate']:.1f}%", (10, y), Config.FONT, 0.5, (255, 165, 0), 1)
        
        # Объединение с кадром
        combined = np.hstack([frame, panel])
        return combined

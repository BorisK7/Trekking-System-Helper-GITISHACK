"""
Модуль для визуализации траекторий, heat map, зон, статистики и скелетов
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
    
    def get_state_color(self, state):
        """Получение цвета для состояния"""
        return Config.STATE_COLORS.get(state, Config.STATE_COLORS['UNKNOWN'])
    
    def draw_trajectories(self, frame, tracker, min_length=5, thickness=4):
        """
        Отрисовка траекторий движения
        """
        for track_id, track in tracker.tracks.items():
            trajectory = track['trajectory']
            
            if len(trajectory) < min_length:
                continue
            
            # Цвет зависит от состояния
            state = track.get('state', 'UNKNOWN')
            color = self.get_state_color(state)
            
            # Отрисовка линии траектории
            points = [(int(x), int(y)) for x, y, _ in trajectory]
            
            # Градиент прозрачности
            for i in range(1, len(points)):
                alpha = 0.3 + (i / len(points)) * 0.7
                thickness_current = max(1, int(thickness * alpha))
                
                # Для статических объектов рисуем пунктир (имитация)
                if state == 'STATIC' and i % 2 == 0:
                    continue
                    
                cv2.line(frame, points[i-1], points[i], color, thickness_current)
            
            # Стрелка направления для MOVABLE/DYNAMIC
            if state in ['MOVABLE', 'DYNAMIC'] and len(points) >= 2:
                self._draw_arrow(frame, points[-2], points[-1], color, thickness)
        
        return frame
    
    def _draw_arrow(self, frame, pt1, pt2, color, thickness):
        """Отрисовка стрелки направления"""
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 5: return
        
        dx /= length
        dy /= length
        
        arrow_len = min(10, length * 0.4)
        arrow_angle = np.pi / 6
        
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
    
    def draw_track_info(self, frame, tracked_objects, tracker, show_speed=True, show_id=True, actions=None, show_state=True):
        """
        Отрисовка информации о треках с поддержкой состояний
        """
        for item in tracked_objects:
            # Распаковка может отличаться в зависимости от трекера
            if len(item) == 8:
                track_id, x1, y1, x2, y2, class_id, conf, state = item
            else:
                track_id, x1, y1, x2, y2, class_id, conf = item
                state = 'UNKNOWN'

            # Цвет рамки зависит от состояния
            color = self.get_state_color(state)
            
            # Толщина зависит от уверенности в состоянии (если есть)
            thickness = 2
            if state == 'STATIC': thickness = 1
            elif state == 'MOVABLE': thickness = 2
            elif state == 'DYNAMIC': thickness = 3
            
            # Рамка
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Информация
            class_name = Config.COCO_CLASSES.get(class_id, 'unknown')
            info_lines = []
            
            if actions and track_id in actions:
                action_label, action_conf = actions[track_id]
                if action_label != "unknown":
                    info_lines.append(f"ACT: {action_label.upper()} ({action_conf:.2f})")
            
            if show_state and state != 'UNKNOWN':
                info_lines.append(f"ST: {state}")
                
            if show_id:
                info_lines.append(f"ID:{track_id}")
            
            info_lines.append(f"{class_name.title()}")
            
            if show_speed and state != 'STATIC':
                speed_px_s = tracker.get_speed(track_id)
                if speed_px_s > 1:
                    info_lines.append(f"{speed_px_s:.1f}px/s")
            
            # Отрисовка текста
            y_offset = int(y1) - 10
            for line in reversed(info_lines):
                bg_color = color
                
                # Специальные цвета для действий
                if "ACT:" in line:
                    if "FALLING" in line: bg_color = (0, 0, 255)
                    elif "RUNNING" in line: bg_color = (0, 165, 255)
                
                (tw, th), _ = cv2.getTextSize(line, Config.FONT, 0.5, 1)
                
                # Фон текста
                cv2.rectangle(
                    frame,
                    (int(x1), y_offset - th - 2),
                    (int(x1) + tw, y_offset),
                    bg_color,
                    -1
                )
                
                # Текст
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

    def draw_skeletons(self, frame, keypoints_dict):
        """Отрисовка скелетов"""
        skeleton_links = [
            (5, 7), (7, 9), (6, 8), (8, 10), # Руки
            (5, 6), (5, 11), (6, 12), (11, 12), # Тело
            (11, 13), (13, 15), (12, 14), (14, 16) # Ноги
        ]
        
        for track_id, kpts in keypoints_dict.items():
            if kpts is None: continue
            
            # Цвет скелета берем стандартный (по ID), чтобы отличался от рамок состояния
            color = self.get_color_for_track(track_id)
            
            for i, (x, y, conf) in enumerate(kpts):
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 3, color, -1)
            
            for p1_idx, p2_idx in skeleton_links:
                if p1_idx < len(kpts) and p2_idx < len(kpts):
                    x1, y1, c1 = kpts[p1_idx]
                    x2, y2, c2 = kpts[p2_idx]
                    if c1 > 0.5 and c2 > 0.5:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        return frame


class HeatMapVisualizer:
    """Визуализация heat map активности"""
    
    @staticmethod
    def overlay_heat_map(frame, heat_map, alpha=0.5):
        if heat_map.shape[:2] != frame.shape[:2]:
            heat_map = cv2.resize(heat_map, (frame.shape[1], frame.shape[0]))
        result = cv2.addWeighted(frame, 1 - alpha, heat_map, alpha, 0)
        return result


class ZoneVisualizer:
    """Визуализация зон интереса и линий подсчета"""
    
    @staticmethod
    def draw_zones(frame, zones, analytics, show_stats=True):
        for zone_id, zone in zones.items():
            polygon = np.array(zone['polygon'], dtype=np.int32)
            count = zone.get('count', 0)
            
            if count == 0: color = (100, 100, 100)
            elif count < 3: color = (0, 255, 0)
            elif count < 6: color = (0, 165, 255)
            else: color = (0, 0, 255)
            
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            cv2.polylines(frame, [polygon], True, color, 2)
            
            if show_stats and count > 0:
                M = cv2.moments(polygon)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(frame, f"Zone {zone_id}: {count}", (cx - 40, cy), Config.FONT, 0.6, (255, 255, 255), 2)
        return frame
    
    @staticmethod
    def draw_counting_lines(frame, counting_lines, show_count=True):
        for line_id, line in counting_lines.items():
            start = tuple(map(int, line['start']))
            end = tuple(map(int, line['end']))
            count = line.get('cross_count', 0)
            
            cv2.line(frame, start, end, (255, 0, 255), 3)
            
            if show_count:
                mid_x = (start[0] + end[0]) // 2
                mid_y = (start[1] + end[1]) // 2
                text = f"Line {line_id}: {count}"
                (tw, th), _ = cv2.getTextSize(text, Config.FONT, 0.7, 2)
                cv2.rectangle(frame, (mid_x - tw//2 - 5, mid_y - th - 5), (mid_x + tw//2 + 5, mid_y + 5), (255, 0, 255), -1)
                cv2.putText(frame, text, (mid_x - tw//2, mid_y), Config.FONT, 0.7, (255, 255, 255), 2)
        return frame


class StatisticsOverlay:
    """Наложение статистики на кадр"""
    
    @staticmethod
    def draw_statistics_panel(frame, tracker, analytics=None, occupancy=None, action_stats=None):
        h, w = frame.shape[:2]
        panel_width = 300
        panel_height = h
        
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        y = 30
        cv2.putText(panel, "TRACKING STATS", (10, y), Config.FONT, 0.6, (255, 255, 255), 2)
        y += 30
        
        stats = tracker.get_statistics()
        cv2.putText(panel, f"Active tracks: {stats['active_tracks']}", (10, y), Config.FONT, 0.5, (200, 200, 200), 1)
        y += 25
        cv2.putText(panel, f"Total created: {stats['total_tracks_created']}", (10, y), Config.FONT, 0.5, (200, 200, 200), 1)
        y += 30
        
        # Легенда состояний
        cv2.putText(panel, "STATE LEGEND", (10, y), Config.FONT, 0.6, (255, 255, 255), 2)
        y += 25
        for state, color in Config.STATE_COLORS.items():
            cv2.circle(panel, (20, y-5), 5, color, -1)
            cv2.putText(panel, state, (35, y), Config.FONT, 0.5, (200, 200, 200), 1)
            y += 20
        y += 10

        if action_stats:
            y += 10
            cv2.putText(panel, "HUMAN ACTIONS", (10, y), Config.FONT, 0.6, (255, 255, 255), 2)
            y += 30
            priority = ['falling', 'running', 'walking', 'sitting', 'standing']
            for act in priority:
                count = action_stats.get(act, 0)
                if count > 0:
                    col = (200, 200, 200)
                    if act == 'falling': col = (0, 0, 255)
                    elif act == 'running': col = (0, 165, 255)
                    cv2.putText(panel, f"{act.title()}: {count}", (10, y), Config.FONT, 0.5, col, 1)
                    y += 25

        combined = np.hstack([frame, panel])
        return combined

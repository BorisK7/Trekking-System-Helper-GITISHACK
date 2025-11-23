"""
Геометрические и математические утилиты для анализа видео
"""

import numpy as np


def calculate_angle(a, b, c):
    """
    Расчет угла между тремя точками (в градусах).
    b - вершина угла.
    
    Args:
        a, b, c: Координаты точек (x, y) или [x, y, score]
    
    Returns:
        angle: Угол в градусах (0-180)
    """
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])
    
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)


def point_in_polygon(point, polygon):
    """
    Проверка нахождения точки в полигоне (Ray Casting Algorithm).
    
    Args:
        point: (x, y)
        polygon: Список точек [(x, y), ...]
    
    Returns:
        True если точка внутри, иначе False
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def point_side_of_line(point, line_start, line_end):
    """
    Определение на какой стороне линии находится точка.
    Использует векторное произведение (cross product).
    
    Args:
        point: (x, y)
        line_start: (x, y) начала линии
        line_end: (x, y) конца линии
        
    Returns:
        'above' или 'below' (относительно вектора линии)
    """
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    d = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    
    return 'above' if d > 0 else 'below'


def boxes_overlap_iou(box1, box2):
    """
    Расчет коэффициента перекрытия (Intersection over Union) или простого перекрытия.
    Здесь реализован расчет доли перекрытия относительно МЕНЬШЕГО бокса
    (для определения вложенности/занятости).
    
    Args:
        box1, box2: (x1, y1, x2, y2)
        
    Returns:
        overlap_ratio: Доля перекрытия (0.0 - 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Координаты пересечения
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Площади боксов
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Берем меньшую площадь для оценки "занятости" (если человек сел на стул, 
    # нас интересует, насколько стул перекрыт человеком)
    smaller_area = min(box1_area, box2_area)
    
    if smaller_area == 0:
        return 0.0
    
    return inter_area / smaller_area


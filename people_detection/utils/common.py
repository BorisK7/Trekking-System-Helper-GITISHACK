"""
Общие утилиты и вспомогательные функции
"""

def unpack_track(track):
    """
    Универсальная распаковка данных трека.
    Поддерживает форматы с состоянием (8 элементов) и без (7 элементов).
    
    Args:
        track: кортеж/список данных трека
        
    Returns:
        tuple: (track_id, x1, y1, x2, y2, class_id, conf, state)
        state будет 'UNKNOWN', если его нет во входных данных.
    """
    if len(track) == 8:
        # track_id, x1, y1, x2, y2, class_id, conf, state
        return track
    elif len(track) == 7:
        # track_id, x1, y1, x2, y2, class_id, conf
        track_id, x1, y1, x2, y2, class_id, conf = track
        return track_id, x1, y1, x2, y2, class_id, conf, 'UNKNOWN'
    else:
        # Fallback для нестандартных форматов - возвращаем как есть + UNKNOWN для недостающих
        # Это защитная мера, но лучше так не делать
        result = list(track)
        while len(result) < 7:
            result.append(0)
        if len(result) == 7:
            result.append('UNKNOWN')
        return tuple(result[:8])


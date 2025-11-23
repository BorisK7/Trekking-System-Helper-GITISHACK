import numpy as np
from .config import Config
from collections import deque

class StateTracker:
    """
    Усовершенствованный трекер с поддержкой состояний объектов
    (STATIC, MOVABLE, DYNAMIC) для уменьшения фликеринга ID.
    """
    
    def __init__(self, max_age_static=100, max_age_movable=50, max_age_dynamic=30):
        self.max_age_static = max_age_static
        self.max_age_movable = max_age_movable
        self.max_age_dynamic = max_age_dynamic
        
        # Треки: {track_id: {data}}
        self.tracks = {}
        self.next_id = 1
        
        # Статистика
        self.total_tracks_created = 0
        
        # Пороги движения (пикселей)
        self.movement_threshold = 10.0 
        self.static_variance_threshold = 5.0

    def update(self, detections, frame_time):
        """
        Обновление трекера новыми детекциями
        
        Args:
            detections: [(x1, y1, x2, y2, class_id, conf), ...]
            frame_time: временная метка
            
        Returns:
            active_tracks: [(track_id, x1, y1, x2, y2, class_id, conf, state), ...]
        """
        
        # 1. Предикт (обновление времени жизни)
        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            track['age'] += 1
            
            # Адаптивное удаление
            max_age = self._get_max_age(track['state'])
            
            if track['age'] > max_age:
                del self.tracks[tid]
        
        # 2. Матчинг (Сопоставление)
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)
        
        # 3. Обновление сматченных треков
        for tid, det_idx in matched:
            self._update_track(tid, detections[det_idx], frame_time)
            
        # 4. Создание новых треков
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx], frame_time)
            
        # 5. Формирование результата
        active_tracks = []
        for tid, track in self.tracks.items():
            # Возвращаем трек, если он подтвержден ('hits' >= min_hits) или только что создан
            # Для STATIC объектов показываем даже если age > 0 (память)
            
            # Пропускаем совсем старые "мертвые" треки для визуализации, 
            # но храним их в памяти для восстановления (re-id логика)
            if track['age'] < 2: # Если трек обновлялся недавно
                x1, y1, x2, y2 = track['bbox']
                active_tracks.append((
                    tid, x1, y1, x2, y2, 
                    track['class_id'], track['conf'], 
                    track['state']
                ))
            elif track['state'] == 'STATIC' and track['age'] < 20:
                # Показываем статические объекты чуть дольше ("призраки"), чтобы не мигали
                x1, y1, x2, y2 = track['bbox']
                # Можно добавить флаг 'lost' или 'ghost'
                active_tracks.append((
                    tid, x1, y1, x2, y2, 
                    track['class_id'], track['conf'], 
                    'UNKNOWN' # Помечаем как неуверенные
                ))
                
        return active_tracks

    def _get_max_age(self, state):
        if state == 'STATIC': return self.max_age_static
        if state == 'MOVABLE': return self.max_age_movable
        return self.max_age_dynamic

    def _match_detections(self, detections):
        """Жадное сопоставление по IoU"""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
            
        track_ids = list(self.tracks.keys())
        
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                track_bbox = self.tracks[tid]['bbox']
                det_bbox = det[:4]
                
                # Проверка класса (не матчим стул с человеком)
                if self.tracks[tid]['class_id'] != det[4]:
                    iou_matrix[i, j] = 0.0
                    continue
                    
                iou = self._calculate_iou(track_bbox, det_bbox)
                iou_matrix[i, j] = iou
                
        # Жадный выбор
        matched = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(track_ids)))
        
        if iou_matrix.size > 0:
            # Сортируем индексы по убыванию IoU
            flat_indices = np.argsort(-iou_matrix.flatten())
            
            for idx in flat_indices:
                i, j = divmod(idx, iou_matrix.shape[1])
                
                if iou_matrix[i, j] < 0.3: # Порог IoU
                    break
                    
                if i in unmatched_tracks and j in unmatched_dets:
                    matched.append((track_ids[i], j))
                    unmatched_tracks.remove(i)
                    unmatched_dets.remove(j)
                    
        return matched, list(unmatched_dets), [track_ids[i] for i in unmatched_tracks]

    def _create_track(self, detection, frame_time):
        x1, y1, x2, y2, cls_id, conf = detection
        
        # Определяем начальное состояние на основе класса
        class_name = Config.COCO_CLASSES.get(cls_id, 'unknown')
        init_state = 'UNKNOWN'
        
        if class_name in Config.DYNAMIC_CATEGORIES:
            init_state = 'DYNAMIC'
        
        self.tracks[self.next_id] = {
            'bbox': (x1, y1, x2, y2),
            'class_id': cls_id,
            'conf': conf,
            'age': 0,
            'hits': 1,
            'state': init_state,
            'history': deque(maxlen=30), # История центров
            'trajectory': [( (x1+x2)/2, (y1+y2)/2, frame_time )],
            'state_conf': 0.5
        }
        self.next_id += 1
        self.total_tracks_created += 1

    def _update_track(self, tid, detection, frame_time):
        x1, y1, x2, y2, cls_id, conf = detection
        track = self.tracks[tid]
        
        # Обновляем бокс и сбрасываем возраст
        track['bbox'] = (x1, y1, x2, y2)
        track['conf'] = conf
        track['age'] = 0
        track['hits'] += 1
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        track['history'].append((cx, cy))
        track['trajectory'].append((cx, cy, frame_time))
        
        # Анализ состояния
        self._analyze_state(track, cls_id)

    def _analyze_state(self, track, cls_id):
        """Анализ движения для определения состояния STATIC/MOVABLE"""
        class_name = Config.COCO_CLASSES.get(cls_id, 'unknown')
        
        # Только для потенциально статичных объектов
        if class_name in Config.DYNAMIC_CATEGORIES:
            track['state'] = 'DYNAMIC'
            return

        if len(track['history']) < 5:
            track['state'] = 'UNKNOWN'
            return
            
        # Расчет дисперсии координат (насколько объект дрожит/движется)
        points = np.array(track['history'])
        variance = np.var(points, axis=0)
        movement = np.mean(variance)
        
        if movement < self.static_variance_threshold:
            track['state'] = 'STATIC'
            track['state_conf'] = min(1.0, track['state_conf'] + 0.05)
        elif movement > self.movement_threshold:
            track['state'] = 'MOVABLE'
            track['state_conf'] = min(1.0, track['state_conf'] + 0.05)
        else:
            # Пограничное состояние, сохраняем предыдущее или UNKNOWN
            pass

    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0: return 0
        return inter_area / union_area
        
    def get_statistics(self):
        return {
            'total_tracks_created': self.total_tracks_created,
            'active_tracks': len(self.tracks)
        }
    
    def get_track_duration(self, tid):
        if tid in self.tracks:
            traj = self.tracks[tid]['trajectory']
            if traj:
                return traj[-1][2] - traj[0][2]
        return 0
        
    def get_total_distance(self, tid):
        dist = 0
        if tid in self.tracks:
            traj = self.tracks[tid]['trajectory']
            for i in range(1, len(traj)):
                p1 = traj[i-1]
                p2 = traj[i]
                dist += ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
        return dist
        
    def get_speed(self, tid):
        if tid in self.tracks:
            traj = self.tracks[tid]['trajectory']
            if len(traj) < 2: return 0
            # Скорость за последнюю секунду (примерно)
            # Берем последние 10 точек
            recent = traj[-10:]
            dist = 0
            for i in range(1, len(recent)):
                p1 = recent[i-1]
                p2 = recent[i]
                dist += ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
            
            duration = recent[-1][2] - recent[0][2]
            if duration > 0:
                return dist / duration
        return 0


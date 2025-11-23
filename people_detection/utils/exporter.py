"""
Модуль для экспорта данных трекинга и аналитики
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

from .analytics import SceneAnalytics
from .common import unpack_track


class DataExporter:
    """Экспорт данных в JSON/CSV для анализа и сравнения с UWB"""
    
    def __init__(self, output_dir: Union[str, Path] = 'output/analytics'):
        """
        Args:
            output_dir: Директория для сохранения файлов
        """
        self.output_dir = Path(output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Ошибка при создании директории {self.output_dir}: {e}")
        
        # Собранные данные
        self.trajectories_data: List[Dict[str, Any]] = []
        self.frame_data: List[Dict[str, Any]] = []
        self.statistics: Dict[str, Any] = {}
    
    def add_frame_data(self, 
                       frame_number: int, 
                       timestamp: float, 
                       tracked_objects: List[Tuple], 
                       analytics: Optional[SceneAnalytics] = None,
                       actions: Optional[Dict[int, Tuple[str, float]]] = None):
        """
        Добавление данных кадра
        """
        from .config import Config
        
        frame_info = {
            'frame': frame_number,
            'timestamp': timestamp,
            'objects': []
        }
        
        for track in tracked_objects:
            track_id, x1, y1, x2, y2, class_id, conf, state = unpack_track(track)
            
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            obj_data = {
                'track_id': track_id,
                'class': Config.COCO_CLASSES.get(class_id, 'unknown'),
                'class_id': class_id,
                'bbox': {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)},
                'center': {'x': float(cx), 'y': float(cy)},
                'confidence': float(conf)
            }
            
            # Добавляем состояние
            if state != 'UNKNOWN':
                obj_data['state'] = state

            # Добавляем действие, если есть
            if actions and track_id in actions:
                action_label, action_conf = actions[track_id]
                obj_data['action'] = {
                    'label': action_label,
                    'confidence': float(action_conf)
                }
            
            frame_info['objects'].append(obj_data)
        
        # Статистика зон
        if analytics:
            zone_stats = analytics.get_zone_statistics()
            if zone_stats:
                frame_info['zones'] = zone_stats
        
        self.frame_data.append(frame_info)
    
    def add_trajectory(self, 
                       track_id: int, 
                       trajectory: List[Tuple], 
                       class_name: str, 
                       metadata: Optional[Dict[str, Any]] = None):
        """
        Добавление траектории объекта
        """
        traj_data = {
            'track_id': track_id,
            'class': class_name,
            'points': [],
            'metadata': metadata or {}
        }
        
        for point in trajectory:
            # Поддержка разного количества параметров в точке
            if len(point) >= 5:
                x, y, t, px, py = point[:5]
                point_data = {
                    'x': float(x),
                    'y': float(y),
                    'timestamp': float(t),
                    'proj_x': float(px),
                    'proj_y': float(py)
                }
            else:
                x, y, t = point[:3]
                point_data = {
                    'x': float(x),
                    'y': float(y),
                    'timestamp': float(t)
                }
            
            traj_data['points'].append(point_data)
        
        self.trajectories_data.append(traj_data)
    
    def set_statistics(self, stats: Dict[str, Any]):
        """Установка общей статистики"""
        self.statistics = stats
    
    def export_trajectories_json(self, filename: str = 'trajectories.json') -> Path:
        """Экспорт траекторий в JSON"""
        output_file = self.output_dir / filename
        
        data = {
            'export_date': datetime.now().isoformat(),
            'total_tracks': len(self.trajectories_data),
            'trajectories': self.trajectories_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def export_trajectories_csv(self, filename: str = 'trajectories.csv') -> Path:
        """Экспорт траекторий в CSV (формат для сравнения с UWB)"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Определяем заголовки на основе данных
            headers = ['track_id', 'class', 'x', 'y', 'timestamp']
            has_projection = False
            
            # Проверяем наличие проекции в первой точке любой траектории
            for traj in self.trajectories_data:
                if traj['points'] and 'proj_x' in traj['points'][0]:
                    headers.extend(['proj_x', 'proj_y'])
                    has_projection = True
                    break
            
            writer.writerow(headers)
            
            for traj in self.trajectories_data:
                track_id = traj['track_id']
                class_name = traj['class']
                
                for point in traj['points']:
                    row = [
                        track_id,
                        class_name,
                        point['x'],
                        point['y'],
                        point['timestamp']
                    ]
                    
                    if has_projection:
                        row.extend([
                            point.get('proj_x', ''),
                            point.get('proj_y', '')
                        ])
                        
                    writer.writerow(row)
        
        return output_file
    
    def export_frame_data_json(self, filename: str = 'frame_data.json') -> Path:
        """Экспорт покадровых данных в JSON"""
        output_file = self.output_dir / filename
        
        data = {
            'export_date': datetime.now().isoformat(),
            'total_frames': len(self.frame_data),
            'frames': self.frame_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def export_statistics_json(self, filename: str = 'statistics.json') -> Path:
        """Экспорт статистики в JSON"""
        output_file = self.output_dir / filename
        
        data = {
            'export_date': datetime.now().isoformat(),
            'statistics': self.statistics
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def export_for_uwb_comparison(self, filename: str = 'uwb_comparison.csv') -> Path:
        """
        Экспорт данных в формате для сравнения с UWB
        Формат совместимый с walking_path.csv из UWB датасета
        """
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Заголовок как в UWB: x, y, z
            writer.writerow(['track_id', 'x', 'y', 'z', 'timestamp', 'class'])
            
            for traj in self.trajectories_data:
                track_id = traj['track_id']
                class_name = traj['class']
                
                for point in traj['points']:
                    writer.writerow([
                        track_id,
                        point['x'],
                        point['y'],
                        0.0,  # z (высота) - по умолчанию 0
                        point['timestamp'],
                        class_name
                    ])
        
        return output_file
    
    def export_all(self, base_filename: str = 'detection_analysis') -> Dict[str, str]:
        """Экспорт всех данных"""
        exported_files = {}
        
        try:
            # Траектории
            exported_files['trajectories_json'] = str(self.export_trajectories_json(
                f'{base_filename}_trajectories.json'
            ))
            exported_files['trajectories_csv'] = str(self.export_trajectories_csv(
                f'{base_filename}_trajectories.csv'
            ))
            
            # Покадровые данные
            if self.frame_data:
                exported_files['frame_data'] = str(self.export_frame_data_json(
                    f'{base_filename}_frames.json'
                ))
            
            # Статистика
            if self.statistics:
                exported_files['statistics'] = str(self.export_statistics_json(
                    f'{base_filename}_stats.json'
                ))
            
            # UWB формат
            exported_files['uwb_format'] = str(self.export_for_uwb_comparison(
                f'{base_filename}_uwb_format.csv'
            ))
            
            # Добавляем копирование видео если требуется (реализовано в main)
            
        except Exception as e:
            print(f"Ошибка при экспорте: {e}")
        
        return exported_files
    
    def create_summary_report(self, filename: str = 'summary.txt') -> Path:
        """Создание текстового отчета"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ОТЧЕТ О ДЕТЕКТИРОВАНИИ И ТРЕКИНГЕ ОБЪЕКТОВ\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Дата экспорта: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Всего кадров обработано: {len(self.frame_data)}\n")
            f.write(f"Всего треков создано: {len(self.trajectories_data)}\n\n")
            
            if self.statistics:
                f.write("СТАТИСТИКА:\n")
                for key, value in self.statistics.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Статистика по классам
            class_counts = {}
            for traj in self.trajectories_data:
                class_name = traj['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if class_counts:
                f.write("РАСПРЕДЕЛЕНИЕ ТРЕКОВ ПО КЛАССАМ:\n")
                for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                    f.write(f"  {class_name}: {count}\n")
        
        return output_file

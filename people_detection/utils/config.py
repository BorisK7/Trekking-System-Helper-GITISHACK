"""
Конфигурация для детектирования людей
"""

import torch


class Config:
    """Конфигурация детектора объектов"""
    
    # Доступные модели YOLOv8
    AVAILABLE_MODELS = {
        'yolov8n': 'yolov8n.pt',  # Nano - самая быстрая
        'yolov8s': 'yolov8s.pt',  # Small
        'yolov8m': 'yolov8m.pt',  # Medium
        'yolov8l': 'yolov8l.pt',  # Large
        'yolov8x': 'yolov8x.pt',  # Extra Large - самая точная
    }
    
    # Модель по умолчанию
    DEFAULT_MODEL = 'yolov8x'
    
    # Все классы COCO dataset (80 классов)
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
        15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
        45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
        55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
        65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
        70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
        75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }
    
    # Класс "person" в COCO dataset
    PERSON_CLASS_ID = 0
    
    # Цвета для разных категорий объектов (BGR формат)
    CATEGORY_COLORS = {
        'person': (0, 255, 0),      # Зеленый - люди
        'vehicle': (0, 165, 255),   # Оранжевый - транспорт
        'animal': (255, 0, 255),    # Фиолетовый - животные
        'furniture': (255, 255, 0), # Голубой - мебель
        'electronics': (0, 255, 255), # Желтый - электроника
        'food': (147, 20, 255),     # Розовый - еда
        'default': (0, 200, 200)    # Желто-зеленый - остальное
    }
    
    # Устройство для вычислений
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Настройки по умолчанию
    DEFAULT_CONF_THRESHOLD = 0.5
    DEFAULT_IOU_THRESHOLD = 0.45
    
    # Цвета для визуализации (BGR формат для OpenCV)
    BOX_COLOR = (0, 255, 0)  # Зеленый
    TEXT_COLOR = (255, 255, 255)  # Белый
    TEXT_BG_COLOR = (0, 200, 0)  # Темно-зеленый
    
    # Параметры шрифта
    FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    BOX_THICKNESS = 2
    
    @classmethod
    def get_device_info(cls):
        """Получить информацию об используемом устройстве"""
        import torch
        
        if not torch.cuda.is_available():
            return "CPU (CUDA не доступна - установите PyTorch с CUDA)"
        
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            cuda_version = torch.version.cuda
            return f"GPU: {gpu_name} ({gpu_memory:.1f} GB) | CUDA {cuda_version}"
        except Exception as e:
            return f"CPU (Ошибка GPU: {e})"
    
    @classmethod
    def get_category_color(cls, class_id):
        """Получить цвет для класса объекта"""
        class_name = cls.COCO_CLASSES.get(class_id, 'unknown')
        
        # Люди
        if class_id == 0:
            return cls.CATEGORY_COLORS['person']
        # Транспорт
        elif class_id in [1, 2, 3, 5, 6, 7, 8]:
            return cls.CATEGORY_COLORS['vehicle']
        # Животные
        elif class_id in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
            return cls.CATEGORY_COLORS['animal']
        # Мебель
        elif class_id in [56, 57, 58, 59, 60, 61]:
            return cls.CATEGORY_COLORS['furniture']
        # Электроника
        elif class_id in [62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 74]:
            return cls.CATEGORY_COLORS['electronics']
        # Еда
        elif class_id in [46, 47, 48, 49, 50, 51, 52, 53, 54, 55]:
            return cls.CATEGORY_COLORS['food']
        else:
            return cls.CATEGORY_COLORS['default']
    
    @classmethod
    def validate_model(cls, model_name):
        """Проверить доступность модели"""
        if model_name not in cls.AVAILABLE_MODELS:
            available = ', '.join(cls.AVAILABLE_MODELS.keys())
            raise ValueError(f"Модель '{model_name}' недоступна. Доступные: {available}")
        return cls.AVAILABLE_MODELS[model_name]
    
    @classmethod
    def parse_classes(cls, classes_str):
        """Парсинг строки с классами в список ID"""
        if not classes_str or classes_str.lower() == 'all':
            return None  # Все классы
        
        try:
            # Попытка парсинга как числа или списка чисел
            class_ids = [int(c.strip()) for c in classes_str.split(',')]
            # Проверка валидности
            for cid in class_ids:
                if cid not in cls.COCO_CLASSES:
                    raise ValueError(f"Неверный ID класса: {cid}")
            return class_ids
        except ValueError:
            # Попытка парсинга как имена классов
            class_names = [c.strip().lower() for c in classes_str.split(',')]
            name_to_id = {v.lower(): k for k, v in cls.COCO_CLASSES.items()}
            class_ids = []
            for name in class_names:
                if name in name_to_id:
                    class_ids.append(name_to_id[name])
                else:
                    raise ValueError(f"Неизвестный класс: {name}")
            return class_ids if class_ids else None

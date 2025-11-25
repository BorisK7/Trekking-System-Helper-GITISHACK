import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

enum TimelineElementType { 
  actor,       // Актеры
  stageDir,    // Механик сцены (слова помрежа)
  mounting,    // Монтировочный цех
  sound,       // Звуковой цех
  light,       // Световой цех
}

class TimelineElement {
  final String id;
  final TimelineElementType type;
  final String name;
  final Duration startTime;
  final Duration endTime;
  final String? subGroup;      // Подгруппа (например, "Монтировщик 1")
  final String? description;   // Дополнительное описание
  final Color color;

  TimelineElement({
    required this.id,
    required this.type,
    required this.name,
    required this.startTime,
    required this.endTime,
    this.subGroup,
    this.description,
    required this.color,
  });

  Duration get duration => endTime - startTime;
}

class TimelineGroup {
  final String name;
  final TimelineElementType type;
  final Color color;
  final List<TimelineElement> elements;
  final List<TimelineSubGroup> subGroups;
  bool isExpanded;

  TimelineGroup({
    required this.name,
    required this.type,
    required this.color,
    this.elements = const [],
    this.subGroups = const [],
    this.isExpanded = true,
  });
}

class TimelineSubGroup {
  final String name;
  final List<TimelineElement> elements;

  TimelineSubGroup({
    required this.name,
    required this.elements,
  });
}

class DashboardState extends ChangeNotifier {
  VideoPlayerController? _videoController;
  VideoPlayerController? get videoController => _videoController;

  VideoPlayerController? _rightVideoController;
  VideoPlayerController? get rightVideoController => _rightVideoController;

  bool get isInitialized => (_videoController?.value.isInitialized ?? false) && 
                           (_rightVideoController?.value.isInitialized ?? false);

  // Player State
  bool get isPlaying => _videoController?.value.isPlaying ?? false;

  Duration get currentPosition => _videoController?.value.position ?? Duration.zero;
  
  Duration get totalDuration => _videoController?.value.duration ?? const Duration(minutes: 1);

  // Selection State
  TimelineElement? _selectedElement;
  TimelineElement? get selectedElement => _selectedElement;

  // Timeline Groups
  final List<TimelineGroup> _timelineGroups = [
    // Актеры
    TimelineGroup(
      name: 'Актеры',
      type: TimelineElementType.actor,
      color: const Color(0xFF4CAF50),
      elements: [
        TimelineElement(
          id: 'actor_1',
          type: TimelineElementType.actor,
          name: 'Медведь',
          startTime: const Duration(seconds: 0),
          endTime: const Duration(seconds: 6),
          color: const Color(0xFF66BB6A),
        ),
        TimelineElement(
          id: 'actor_2',
          type: TimelineElementType.actor,
          name: 'Волшебник',
          startTime: const Duration(seconds: 0),
          endTime: const Duration(seconds: 34),
          color: const Color(0xFF81C784),
        ),
        TimelineElement(
          id: 'actor_3',
          type: TimelineElementType.actor,
          name: 'Хозяйка',
          startTime: const Duration(seconds: 0),
          endTime: const Duration(seconds: 34),
          color: const Color(0xFFA5D6A7),
        ),
      ],
    ),
    
    // Механик сцены
    TimelineGroup(
      name: 'Механик сцены',
      type: TimelineElementType.stageDir,
      color: const Color(0xFFFF9800),
      elements: [
        TimelineElement(
          id: 'stage_1',
          type: TimelineElementType.stageDir,
          name: 'Опустить колонны',
          startTime: const Duration(seconds: 3),
          endTime: const Duration(seconds: 4),
          color: const Color(0xFFFFB74D),
        ),
      ],
    ),
    
    // Монтировочный цех
    TimelineGroup(
      name: 'Монтировочный цех',
      type: TimelineElementType.mounting,
      color: const Color(0xFF9C27B0),
      subGroups: [
        TimelineSubGroup(
          name: 'Монтировщик 1',
          elements: [
            TimelineElement(
              id: 'mount_1',
              type: TimelineElementType.mounting,
              name: 'Запуск поезда',
              startTime: const Duration(seconds: 29),
              endTime: const Duration(seconds: 30),
              subGroup: 'Монтировщик 1',
              description: 'левый карман',
              color: const Color(0xFFBA68C8),
            ),
          ],
        ),
        TimelineSubGroup(
          name: 'Монтировщик 2',
          elements: [
            TimelineElement(
              id: 'mount_2',
              type: TimelineElementType.mounting,
              name: 'Прием поезда',
              startTime: const Duration(seconds: 34),
              endTime: const Duration(seconds: 35),
              subGroup: 'Монтировщик 2',
              description: 'правый карман',
              color: const Color(0xFFCE93D8),
            ),
          ],
        ),
      ],
    ),
    
    // Звуковой цех
    TimelineGroup(
      name: 'Звуковой цех',
      type: TimelineElementType.sound,
      color: const Color(0xFF2196F3),
      elements: [
        TimelineElement(
          id: 'sound_1',
          type: TimelineElementType.sound,
          name: 'Act1_Love_song.mp3',
          startTime: const Duration(seconds: 2),
          endTime: const Duration(seconds: 34),
          color: const Color(0xFF64B5F6),
        ),
        TimelineElement(
          id: 'sound_2',
          type: TimelineElementType.sound,
          name: 'Микрофон 1 (mic1)',
          startTime: const Duration(seconds: 0),
          endTime: const Duration(seconds: 6),
          color: const Color(0xFF42A5F5),
        ),
        TimelineElement(
          id: 'sound_3',
          type: TimelineElementType.sound,
          name: 'Микрофон 2 (mic2)',
          startTime: const Duration(seconds: 0),
          endTime: const Duration(seconds: 34),
          color: const Color(0xFF90CAF9),
        ),
        TimelineElement(
          id: 'sound_4',
          type: TimelineElementType.sound,
          name: 'Микрофон 3 (mic3)',
          startTime: const Duration(seconds: 0),
          endTime: const Duration(seconds: 34),
          color: const Color(0xFFBBDEFB),
        ),
      ],
    ),
    
    // Световой цех
    TimelineGroup(
      name: 'Световой цех',
      type: TimelineElementType.light,
      color: const Color(0xFFFFEB3B),
      elements: [
        TimelineElement(
          id: 'light_1',
          type: TimelineElementType.light,
          name: 'PS 2кВТ – 2,6,8,12,16 (L201)',
          startTime: const Duration(seconds: 0),
          endTime: const Duration(seconds: 1),
          color: const Color(0xFFFFF176),
        ),
        TimelineElement(
          id: 'light_2',
          type: TimelineElementType.light,
          name: 'PS 1кВт – 20 (L201)',
          startTime: const Duration(seconds: 4),
          endTime: const Duration(seconds: 5),
          description: 'каждый своя колонна, в основание колонны самый узкий луч',
          color: const Color(0xFFFFF59D),
        ),
      ],
    ),
  ];
  
  List<TimelineGroup> get timelineGroups => _timelineGroups;

  void toggleGroupExpanded(int index) {
    _timelineGroups[index].isExpanded = !_timelineGroups[index].isExpanded;
    notifyListeners();
  }

  Future<void> initializeVideo(String assetPath) async {
    _videoController = VideoPlayerController.asset(assetPath);
    _rightVideoController = VideoPlayerController.asset('assets/right.mp4');

    try {
      await Future.wait([
        _videoController!.initialize(),
        _rightVideoController!.initialize(),
      ]);

      // Don't loop the video so it behaves like an editor
      await _videoController!.setLooping(false);
      await _rightVideoController!.setLooping(false);

      // Listen to video updates to refresh UI (playhead)
      _videoController!.addListener(_videoListener);
      notifyListeners();
    } catch (e) {
      debugPrint('Error initializing video: $e');
    }
  }

  void _videoListener() {
    // Notify listeners on position change (for timeline) and state change
    notifyListeners();
  }

  Future<void> togglePlay() async {
    if (_videoController == null || _rightVideoController == null) return;
    
    if (_videoController!.value.isPlaying) {
      await Future.wait([
        _videoController!.pause(),
        _rightVideoController!.pause(),
      ]);
    } else {
      // If video is at the end, restart from beginning
      if (_videoController!.value.position >= _videoController!.value.duration) {
        await seekTo(Duration.zero);
      }
      await Future.wait([
        _videoController!.play(),
        _rightVideoController!.play(),
      ]);
    }
    notifyListeners();
  }

  Future<void> seekTo(Duration position) async {
    if (_videoController == null || _rightVideoController == null) return;
    await Future.wait([
      _videoController!.seekTo(position),
      _rightVideoController!.seekTo(position),
    ]);
    // notifyListeners() will be called by the listener
  }

  void selectElement(TimelineElement? element) {
    _selectedElement = element;
    notifyListeners();
  }

  @override
  void dispose() {
    _videoController?.removeListener(_videoListener);
    _videoController?.dispose();
    _rightVideoController?.dispose();
    super.dispose();
  }
}

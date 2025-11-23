import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

enum EditorElementType { video, audio, text, filter }

class EditorElement {
  final String id;
  final EditorElementType type;
  final String name;
  final Duration startTime;
  final Duration duration;

  EditorElement({
    required this.id,
    required this.type,
    required this.name,
    required this.startTime,
    required this.duration,
  });
}

class DashboardState extends ChangeNotifier {
  VideoPlayerController? _videoController;
  VideoPlayerController? get videoController => _videoController;
  bool get isInitialized => _videoController?.value.isInitialized ?? false;

  // Player State
  bool get isPlaying => _videoController?.value.isPlaying ?? false;

  Duration get currentPosition => _videoController?.value.position ?? Duration.zero;
  
  Duration get totalDuration => _videoController?.value.duration ?? const Duration(minutes: 1);

  // Selection State
  EditorElement? _selectedElement;
  EditorElement? get selectedElement => _selectedElement;

  // Project State
  final List<EditorElement> _elements = [
    EditorElement(
      id: '1',
      type: EditorElementType.video,
      name: 'Video Clip 1',
      startTime: Duration.zero,
      duration: const Duration(seconds: 10),
    ),
    EditorElement(
      id: '2',
      type: EditorElementType.audio,
      name: 'Background Music',
      startTime: Duration.zero,
      duration: const Duration(minutes: 1),
    ),
  ];
  List<EditorElement> get elements => _elements;

  Future<void> initializeVideo(String assetPath) async {
    _videoController = VideoPlayerController.asset(assetPath);
    try {
      await _videoController!.initialize();
      // Don't loop the video so it behaves like an editor
      await _videoController!.setLooping(false);
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
    if (_videoController == null) return;
    
    if (_videoController!.value.isPlaying) {
      await _videoController!.pause();
    } else {
      // If video is at the end, restart from beginning
      if (_videoController!.value.position >= _videoController!.value.duration) {
        await _videoController!.seekTo(Duration.zero);
      }
      await _videoController!.play();
    }
    notifyListeners();
  }

  Future<void> seekTo(Duration position) async {
    if (_videoController == null) return;
    await _videoController!.seekTo(position);
    // notifyListeners() will be called by the listener
  }

  void selectElement(EditorElement? element) {
    _selectedElement = element;
    notifyListeners();
  }
  
  void addElement(EditorElement element) {
    _elements.add(element);
    notifyListeners();
  }

  @override
  void dispose() {
    _videoController?.removeListener(_videoListener);
    _videoController?.dispose();
    super.dispose();
  }
}

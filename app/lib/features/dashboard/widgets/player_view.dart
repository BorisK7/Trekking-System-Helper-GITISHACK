import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import '../models/dashboard_state.dart';

class PlayerView extends StatelessWidget {
  final DashboardState state;

  const PlayerView({super.key, required this.state});

  @override
  Widget build(BuildContext context) {
    return DragTarget<String>(
      builder: (context, candidateData, rejectedData) {
        return Container(
          decoration: BoxDecoration(
            color: const Color(0xFF1E1E1E),
            borderRadius: BorderRadius.circular(4),
            border: Border.all(color: const Color(0xFF3C3C3C), width: 1),
          ),
          child: Stack(
            alignment: Alignment.center,
            children: [
              if (state.isInitialized && state.videoController != null)
                Center(
                  child: AspectRatio(
                    aspectRatio: state.videoController!.value.aspectRatio,
                    child: VideoPlayer(state.videoController!),
                  ),
                )
              else
                const Center(
                  child: CircularProgressIndicator(
                    color: Color(0xFF4CAF50),
                    strokeWidth: 2,
                  ),
                ),

              // Play/Pause overlay control
              if (state.isInitialized)
                GestureDetector(
                  onTap: state.togglePlay,
                  behavior: HitTestBehavior.opaque,
                  child: Center(
                    child: AnimatedOpacity(
                      opacity: state.isPlaying ? 0.0 : 1.0,
                      duration: const Duration(milliseconds: 200),
                      child: Container(
                        decoration: BoxDecoration(
                          color: const Color(0xFF2D2D30).withOpacity(0.8),
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: const Color(0xFF4CAF50),
                            width: 2,
                          ),
                        ),
                        padding: const EdgeInsets.all(16),
                        child: const Icon(
                          Icons.play_arrow,
                          size: 64,
                          color: Color(0xFF4CAF50),
                        ),
                      ),
                    ),
                  ),
                ),

              // Drag highlight
              if (candidateData.isNotEmpty)
                Container(
                  decoration: BoxDecoration(
                    border: Border.all(color: const Color(0xFF4CAF50), width: 4),
                    color: const Color(0xFF4CAF50).withOpacity(0.2),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: const Center(
                     child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.file_upload, size: 64, color: Colors.white),
                        SizedBox(height: 16),
                        Text(
                          'Перетащите для замены',
                          style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold),
                        ),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        );
      },
      onAcceptWithDetails: (details) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Файл загружен: ${details.data}'),
            backgroundColor: const Color(0xFF2D2D30),
          ),
        );
      },
    );
  }
}

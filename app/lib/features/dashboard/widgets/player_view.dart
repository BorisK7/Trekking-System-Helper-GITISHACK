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
          color: Colors.black87,
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
                  child: CircularProgressIndicator(color: Colors.white),
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
                          color: Colors.black45,
                          shape: BoxShape.circle,
                        ),
                        padding: const EdgeInsets.all(16),
                        child: const Icon(
                          Icons.play_arrow,
                          size: 64,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ),
                ),

              // Drag highlight
              if (candidateData.isNotEmpty)
                Container(
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.blueAccent, width: 4),
                    color: Colors.blueAccent.withOpacity(0.2),
                  ),
                  child: const Center(
                     child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.file_upload, size: 64, color: Colors.white),
                        SizedBox(height: 16),
                        Text(
                          'Drop to replace',
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
          SnackBar(content: Text('File dropped: ${details.data}')),
        );
      },
    );
  }
}

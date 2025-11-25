import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import '../models/dashboard_state.dart';
import 'editor_app_bar.dart';

class CanvasVideoPlayer extends StatelessWidget {
  final DashboardState state;
  final OverlayType overlayType;
  final VoidCallback onCloseOverlay;

  const CanvasVideoPlayer({
    super.key,
    required this.state,
    required this.overlayType,
    required this.onCloseOverlay,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: const BoxConstraints.expand(),
      color: const Color(0xFFAAACB0),
      child: Stack(
        children: [
          // Main Video Layer with InteractiveViewer
          ClipRect(
            child: InteractiveViewer(
              minScale: 0.1,
              maxScale: 5.0,
              boundaryMargin: const EdgeInsets.all(double.infinity),
              child: state.isInitialized && state.rightVideoController != null
                  ? AspectRatio(
                      aspectRatio: state.rightVideoController!.value.aspectRatio,
                      child: VideoPlayer(state.rightVideoController!),
                    )
                  : const Center(child: CircularProgressIndicator()),
            ),
          ),
          
          // Overlay Layer
          if (overlayType != OverlayType.none)
            _buildOverlay(context),
        ],
      ),
    );
  }

  Widget _buildOverlay(BuildContext context) {
    return Positioned.fill(
      child: Stack(
        children: [
          // Overlay Image with InteractiveViewer
          ClipRect(
            child: InteractiveViewer(
              minScale: 0.1,
              maxScale: 5.0,
              boundaryMargin: const EdgeInsets.all(double.infinity),
              child: _getOverlayContent(),
            ),
          ),
          
          // Close Button
          Positioned(
            top: 12,
            right: 12,
            child: _buildCloseButton(context),
          ),
          
          // Overlay Label
          Positioned(
            top: 12,
            left: 12,
            child: _buildOverlayLabel(),
          ),
        ],
      ),
    );
  }

  Widget _getOverlayContent() {
    switch (overlayType) {
      case OverlayType.light:
        return Image.asset(
          'assets/light.jpeg',
          fit: BoxFit.contain,
          errorBuilder: (context, error, stackTrace) {
            return Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(
                    Icons.broken_image_outlined,
                    size: 48,
                    color: Colors.white54,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Не удалось загрузить изображение',
                    style: TextStyle(
                      color: Colors.white54,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            );
          },
        );
      case OverlayType.none:
        return const SizedBox();
    }
  }

  Widget _buildCloseButton(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onCloseOverlay,
        borderRadius: BorderRadius.circular(20),
        child: Container(
          width: 32,
          height: 32,
          decoration: BoxDecoration(
            color: const Color(0xFF2D2D30).withOpacity(0.9),
            shape: BoxShape.circle,
            border: Border.all(
              color: const Color(0xFF5A5A5A),
              width: 1,
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.3),
                blurRadius: 8,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: const Icon(
            Icons.close,
            size: 18,
            color: Colors.white70,
          ),
        ),
      ),
    );
  }

  Widget _buildOverlayLabel() {
    String label;
    IconData icon;
    Color accentColor;
    
    switch (overlayType) {
      case OverlayType.light:
        label = 'Свет';
        icon = Icons.lightbulb_outline;
        accentColor = const Color(0xFFFFEB3B);
        break;
      case OverlayType.none:
        return const SizedBox();
    }
    
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0xFF2D2D30).withOpacity(0.9),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(
          color: accentColor.withOpacity(0.5),
          width: 1,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.3),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            size: 14,
            color: accentColor,
          ),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              color: accentColor,
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}

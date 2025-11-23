import 'package:flutter/material.dart';
import '../models/dashboard_state.dart';

class TimelinePanel extends StatelessWidget {
  final DashboardState state;

  const TimelinePanel({super.key, required this.state});

  @override
  Widget build(BuildContext context) {
    const double trackHeight = 60.0;
    const double headerWidth = 40.0;
    const double pixelsPerSecond = 20.0; // Scale
    
    final Duration totalDuration = state.isInitialized 
        ? state.totalDuration 
        : const Duration(minutes: 5);
        
    final double totalWidth = totalDuration.inSeconds * pixelsPerSecond + 100; // + buffer

    return Container(
      color: Colors.white, 
      child: Column(
        children: [
          // Toolbar
          Container(
            height: 40,
            color: Colors.grey[200],
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: Row(
              children: [
                // Play/Pause Button
                IconButton(
                  icon: Icon(
                    state.isPlaying ? Icons.pause : Icons.play_arrow,
                    color: Colors.black87,
                  ),
                  onPressed: () => state.togglePlay(),
                ),
                // Stop Button
                IconButton(
                  icon: const Icon(Icons.stop, color: Colors.black87),
                  onPressed: () async {
                    await state.seekTo(Duration.zero);
                    if (state.isPlaying) {
                      await state.togglePlay(); // pause
                    }
                  },
                ),
                const VerticalDivider(width: 20, indent: 8, endIndent: 8),
                IconButton(
                  icon: const Icon(Icons.undo, size: 20, color: Colors.black87),
                  onPressed: () {},
                ),
                IconButton(
                  icon: const Icon(Icons.redo, size: 20, color: Colors.black87),
                  onPressed: () {},
                ),
                const Spacer(),
                Text(
                  _formatDuration(state.currentPosition),
                  style: const TextStyle(color: Colors.black87),
                ),
                const Spacer(),
                IconButton(
                  icon: const Icon(Icons.zoom_out, size: 20, color: Colors.black87),
                  onPressed: () {},
                ),
                IconButton(
                  icon: const Icon(Icons.zoom_in, size: 20, color: Colors.black87),
                  onPressed: () {},
                ),
              ],
            ),
          ),
          
          Expanded(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Track Headers (Fixed)
                Container(
                  width: headerWidth,
                  color: Colors.grey[100],
                  child: Column(
                    children: [
                      const SizedBox(height: 20), // Ruler offset
                      _buildTrackHeader(Icons.videocam, trackHeight),
                      _buildTrackHeader(Icons.audiotrack, trackHeight),
                      _buildTrackHeader(Icons.text_fields, trackHeight),
                    ],
                  ),
                ),

                // Scrollable Timeline Area
                Expanded(
                  child: SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    physics: const ClampingScrollPhysics(),
                    child: SizedBox(
                      width: totalWidth,
                      child: Stack(
                        children: [
                          // 1. Background Tracks Layer
                          Column(
                            children: [
                              // Ruler
                              SizedBox(
                                height: 20,
                                width: totalWidth,
                                child: CustomPaint(
                                  painter: RulerPainter(),
                                  size: Size(totalWidth, 20),
                                ),
                              ),
                              // Tracks
                              _buildTrackBg(trackHeight),
                              _buildTrackBg(trackHeight),
                              _buildTrackBg(trackHeight),
                            ],
                          ),

                          // 2. Clips Layer (Interactive)
                          Positioned.fill(
                            top: 20, // Offset for ruler
                            child: Column(
                              children: [
                                _buildTrackClips(state.elements.where((e) => e.type == EditorElementType.video).toList(), trackHeight, pixelsPerSecond),
                                _buildTrackClips(state.elements.where((e) => e.type == EditorElementType.audio).toList(), trackHeight, pixelsPerSecond),
                                _buildTrackClips(state.elements.where((e) => e.type == EditorElementType.text).toList(), trackHeight, pixelsPerSecond),
                              ],
                            ),
                          ),

                          // 3. Scrubbing / Seeking Layer (Transparent overlay for drag)
                          // Placing this *behind* clips if we want clips to be selectable, 
                          // or *above* if we want scrubbing everywhere. 
                          // Usually scrubbing is done on the Ruler or empty space.
                          // Let's put it everywhere but let clips capture taps if needed.
                          // Actually, easiest for "moving playhead with mouse" is to allow dragging on the Ruler area 
                          // or allow dragging anywhere if not hitting a clip.
                          
                          // Let's make the Ruler draggable for scrubbing
                          Positioned(
                             top: 0,
                             left: 0,
                             right: 0,
                             height: 20, // Ruler height
                             child: GestureDetector(
                               behavior: HitTestBehavior.opaque,
                               onHorizontalDragUpdate: (details) {
                                  final double dragX = details.localPosition.dx;
                                  final double seconds = dragX / pixelsPerSecond;
                                  final clampedSeconds = seconds.clamp(0.0, totalDuration.inSeconds.toDouble());
                                  state.seekTo(Duration(milliseconds: (clampedSeconds * 1000).round()));
                               },
                               onTapUp: (details) {
                                  final double tapX = details.localPosition.dx;
                                  final double seconds = tapX / pixelsPerSecond;
                                  state.seekTo(Duration(milliseconds: (seconds * 1000).round()));
                               },
                             ),
                          ),

                          // 4. Playhead (Visual indicator)
                          Positioned(
                            left: state.currentPosition.inMilliseconds / 1000 * pixelsPerSecond,
                            top: 0,
                            bottom: 0,
                            child: IgnorePointer(
                              child: Container(
                                width: 2,
                                color: Colors.redAccent,
                                child: Column(
                                  children: [
                                    Icon(Icons.arrow_drop_down, color: Colors.redAccent, size: 16),
                                  ],
                                  ),
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTrackHeader(IconData icon, double height) {
    return Container(
      height: height,
      width: 40,
      margin: const EdgeInsets.symmetric(vertical: 1),
      color: Colors.grey[300],
      child: Center(
        child: Icon(icon, size: 20, color: Colors.black54),
      ),
    );
  }

  Widget _buildTrackBg(double height) {
    return Container(
      height: height,
      margin: const EdgeInsets.symmetric(vertical: 1),
      color: Colors.grey[50], 
    );
  }

  Widget _buildTrackClips(List<EditorElement> clips, double height, double pixelsPerSecond) {
    return Container(
      height: height,
      margin: const EdgeInsets.symmetric(vertical: 1),
      // No background color here, transparent so clicks pass through to background if no clip
      child: Stack(
        children: clips.map((clip) {
             final double left = clip.startTime.inSeconds * pixelsPerSecond;
             final double width = clip.duration.inSeconds * pixelsPerSecond;
             
             return Positioned(
               left: left,
               width: width > 10 ? width : 10,
               top: 4,
               bottom: 4,
               child: GestureDetector(
                 onTap: () {
                   state.selectElement(clip);
                 },
                 child: Container(
                   decoration: BoxDecoration(
                     color: Colors.blueAccent.withOpacity(0.8),
                     borderRadius: BorderRadius.circular(4),
                     border: state.selectedElement?.id == clip.id 
                         ? Border.all(color: Colors.orangeAccent, width: 2)
                         : null,
                     boxShadow: [
                       BoxShadow(color: Colors.black12, blurRadius: 2, offset: const Offset(0, 1)),
                     ],
                   ),
                   child: Center(
                     child: Text(
                       clip.name,
                       style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold),
                       overflow: TextOverflow.ellipsis,
                     ),
                   ),
                 ),
               ),
             );
          }).toList(),
      ),
    );
  }

  String _formatDuration(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, "0");
    String twoDigitMinutes = twoDigits(duration.inMinutes.remainder(60));
    String twoDigitSeconds = twoDigits(duration.inSeconds.remainder(60));
    return "$twoDigitMinutes:$twoDigitSeconds";
  }
}

class RulerPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.black26
      ..strokeWidth = 1;

    for (double i = 0; i < size.width; i += 10) {
      double height = (i % 100 == 0) ? 10 : 5;
      canvas.drawLine(Offset(i, 0), Offset(i, height), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

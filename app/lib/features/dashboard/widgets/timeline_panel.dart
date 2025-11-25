import 'package:flutter/material.dart';
import '../models/dashboard_state.dart';

class TimelinePanel extends StatelessWidget {
  final DashboardState state;

  const TimelinePanel({super.key, required this.state});

  @override
  Widget build(BuildContext context) {
    const double trackHeight = 32.0;
    const double groupHeaderHeight = 28.0;
    const double subGroupHeaderHeight = 24.0;
    const double headerWidth = 200.0;
    
    final Duration totalDuration = state.isInitialized 
        ? state.totalDuration 
        : const Duration(seconds: 40);
        
    final int totalMilliseconds = totalDuration.inMilliseconds > 0 ? totalDuration.inMilliseconds : 1000;
    
    return Container(
      color: const Color(0xFF2D2D30), 
      child: Column(
        children: [
          // Toolbar
          Container(
            height: 40,
            color: const Color(0xFF3C3C3C),
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: Row(
              children: [
                // Play/Pause Button
                IconButton(
                  icon: Icon(
                    state.isPlaying ? Icons.pause : Icons.play_arrow,
                    color: Colors.white,
                  ),
                  onPressed: () => state.togglePlay(),
                ),
                // Stop Button
                IconButton(
                  icon: const Icon(Icons.stop, color: Colors.white),
                  onPressed: () async {
                    await state.seekTo(Duration.zero);
                    if (state.isPlaying) {
                      await state.togglePlay();
                    }
                  },
                ),
                const VerticalDivider(width: 20, indent: 8, endIndent: 8, color: Colors.white24),
                IconButton(
                  icon: const Icon(Icons.undo, size: 20, color: Colors.white70),
                  onPressed: () {},
                ),
                IconButton(
                  icon: const Icon(Icons.redo, size: 20, color: Colors.white70),
                  onPressed: () {},
                ),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                  decoration: BoxDecoration(
                    color: const Color(0xFF1E1E1E),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    _formatDuration(state.currentPosition),
                    style: const TextStyle(
                      color: Colors.white,
                      fontFamily: 'monospace',
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                const Spacer(),
                IconButton(
                  icon: const Icon(Icons.zoom_out, size: 20, color: Colors.white70),
                  onPressed: () {},
                ),
                IconButton(
                  icon: const Icon(Icons.zoom_in, size: 20, color: Colors.white70),
                  onPressed: () {},
                ),
              ],
            ),
          ),
          
          Expanded(
            child: Stack(
              children: [
                Column(
                  children: [
                    // Ruler
                    SizedBox(
                      height: 24,
                      width: double.infinity,
                      child: Row(
                        children: [
                          Container(
                            width: headerWidth,
                            color: const Color(0xFF252526),
                          ),
                          Expanded(
                            child: LayoutBuilder(
                              builder: (context, constraints) {
                                return CustomPaint(
                                  painter: RulerPainter(totalMilliseconds: totalMilliseconds),
                                  size: Size(constraints.maxWidth, 24),
                                );
                              }
                            ),
                          ),
                        ],
                      ),
                    ),
                    
                    // Scrollable Tracks Area
                    Expanded(
                      child: SingleChildScrollView(
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            // Headers Column
                            SizedBox(
                              width: headerWidth,
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.stretch,
                                children: _buildGroupHeaders(
                                  state.timelineGroups,
                                  groupHeaderHeight,
                                  subGroupHeaderHeight,
                                  trackHeight,
                                ),
                              ),
                            ),
                            
                            // Tracks Content Column
                            Expanded(
                              child: LayoutBuilder(
                                builder: (context, constraints) {
                                  final double viewWidth = constraints.maxWidth;
                                  final double pixelsPerMs = viewWidth / totalMilliseconds;
                                  
                                  return Column(
                                    crossAxisAlignment: CrossAxisAlignment.stretch,
                                    children: _buildGroupTracks(
                                      state.timelineGroups,
                                      groupHeaderHeight,
                                      subGroupHeaderHeight,
                                      trackHeight,
                                      pixelsPerMs,
                                      viewWidth,
                                    ),
                                  );
                                }
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),

                // Playhead Overlay
                Positioned.fill(
                  child: LayoutBuilder(
                    builder: (context, constraints) {
                      final double availableWidth = constraints.maxWidth - headerWidth;
                      if (availableWidth <= 0) return const SizedBox();
                      
                      final double pixelsPerMs = availableWidth / totalMilliseconds;
                      final double playheadLeft = headerWidth + (state.currentPosition.inMilliseconds * pixelsPerMs);
                      
                      return Stack(
                        children: [
                          // Touch area for scrubbing
                          Positioned(
                            left: headerWidth,
                            top: 0,
                            right: 0,
                            bottom: 0,
                            child: GestureDetector(
                              behavior: HitTestBehavior.translucent,
                              onHorizontalDragUpdate: (details) {
                                final double dragX = details.localPosition.dx;
                                final double progress = dragX / availableWidth;
                                final int ms = (progress * totalMilliseconds).round();
                                final clampedMs = ms.clamp(0, totalMilliseconds);
                                state.seekTo(Duration(milliseconds: clampedMs));
                              },
                              onTapUp: (details) {
                                final double tapX = details.localPosition.dx;
                                final double progress = tapX / availableWidth;
                                final int ms = (progress * totalMilliseconds).round();
                                state.seekTo(Duration(milliseconds: ms));
                              },
                            ),
                          ),
                          
                          // Playhead Line
                          Positioned(
                            left: playheadLeft,
                            top: 0,
                            bottom: 0,
                            width: 2,
                            child: IgnorePointer(
                              child: Container(
                                color: Colors.redAccent,
                                child: const Column(
                                  children: [
                                    Icon(Icons.arrow_drop_down, color: Colors.redAccent, size: 16),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        ],
                      );
                    }
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  List<Widget> _buildGroupHeaders(
    List<TimelineGroup> groups,
    double groupHeaderHeight,
    double subGroupHeaderHeight,
    double trackHeight,
  ) {
    final List<Widget> widgets = [];
    
    for (int i = 0; i < groups.length; i++) {
      final group = groups[i];
      
      // Group Header
      widgets.add(
        GestureDetector(
          onTap: () => state.toggleGroupExpanded(i),
          child: Container(
            height: groupHeaderHeight,
            padding: const EdgeInsets.symmetric(horizontal: 8),
            decoration: BoxDecoration(
              color: group.color.withOpacity(0.3),
              border: Border(
                left: BorderSide(color: group.color, width: 4),
                bottom: BorderSide(color: Colors.black26, width: 1),
              ),
            ),
            child: Row(
              children: [
                Icon(
                  group.isExpanded ? Icons.expand_more : Icons.chevron_right,
                  color: Colors.white70,
                  size: 18,
                ),
                const SizedBox(width: 4),
                Expanded(
                  child: Text(
                    group.name,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
          ),
        ),
      );
      
      if (group.isExpanded) {
        // SubGroups
        for (final subGroup in group.subGroups) {
          // SubGroup Header
          widgets.add(
            Container(
              height: subGroupHeaderHeight,
              padding: const EdgeInsets.only(left: 24, right: 8),
              decoration: BoxDecoration(
                color: const Color(0xFF333337),
                border: const Border(
                  bottom: BorderSide(color: Colors.black26, width: 1),
                ),
              ),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  subGroup.name,
                  style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 11,
                    fontWeight: FontWeight.w500,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ),
          );
          
          // SubGroup Elements
          for (final element in subGroup.elements) {
            widgets.add(_buildElementHeader(element, trackHeight));
          }
        }
        
        // Direct Elements (not in subgroups)
        for (final element in group.elements) {
          widgets.add(_buildElementHeader(element, trackHeight));
        }
      }
    }
    
    return widgets;
  }

  Widget _buildElementHeader(TimelineElement element, double height) {
    return Container(
      height: height,
      padding: const EdgeInsets.only(left: 16, right: 8),
      decoration: const BoxDecoration(
        color: Color(0xFF252526),
        border: Border(
          bottom: BorderSide(color: Colors.black26, width: 1),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: element.color,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              element.name,
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 11,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  List<Widget> _buildGroupTracks(
    List<TimelineGroup> groups,
    double groupHeaderHeight,
    double subGroupHeaderHeight,
    double trackHeight,
    double pixelsPerMs,
    double viewWidth,
  ) {
    final List<Widget> widgets = [];
    
    for (final group in groups) {
      // Group Header Track (empty space)
      widgets.add(
        Container(
          height: groupHeaderHeight,
          color: const Color(0xFF1E1E1E),
        ),
      );
      
      if (group.isExpanded) {
        // SubGroups
        for (final subGroup in group.subGroups) {
          // SubGroup Header Track
          widgets.add(
            Container(
              height: subGroupHeaderHeight,
              color: const Color(0xFF252526),
            ),
          );
          
          // SubGroup Elements
          for (final element in subGroup.elements) {
            widgets.add(_buildElementTrack(element, trackHeight, pixelsPerMs, viewWidth));
          }
        }
        
        // Direct Elements
        for (final element in group.elements) {
          widgets.add(_buildElementTrack(element, trackHeight, pixelsPerMs, viewWidth));
        }
      }
    }
    
    return widgets;
  }

  Widget _buildElementTrack(
    TimelineElement element,
    double height,
    double pixelsPerMs,
    double viewWidth,
  ) {
    final double left = element.startTime.inMilliseconds * pixelsPerMs;
    final double width = element.duration.inMilliseconds * pixelsPerMs;
    
    return Container(
      height: height,
      width: viewWidth,
      decoration: const BoxDecoration(
        color: Color(0xFF1E1E1E),
        border: Border(
          bottom: BorderSide(color: Colors.black26, width: 1),
        ),
      ),
      child: Stack(
        children: [
          Positioned(
            left: left,
            width: width > 20 ? width : 20,
            top: 4,
            bottom: 4,
            child: Tooltip(
              message: element.description ?? element.name,
              child: Container(
                decoration: BoxDecoration(
                  color: element.color,
                  borderRadius: BorderRadius.circular(3),
                  boxShadow: [
                    BoxShadow(
                      color: element.color.withOpacity(0.3),
                      blurRadius: 4,
                      offset: const Offset(0, 1),
                    ),
                  ],
                ),
                padding: const EdgeInsets.symmetric(horizontal: 6),
                child: Center(
                  child: Text(
                    _formatTimeRange(element.startTime, element.endTime),
                    style: const TextStyle(
                      color: Colors.black87,
                      fontSize: 9,
                      fontWeight: FontWeight.w500,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _formatDuration(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, "0");
    String twoDigitMinutes = twoDigits(duration.inMinutes.remainder(60));
    String twoDigitSeconds = twoDigits(duration.inSeconds.remainder(60));
    return "$twoDigitMinutes:$twoDigitSeconds";
  }

  String _formatTimeRange(Duration start, Duration end) {
    return '${_formatDuration(start)} – ${_formatDuration(end)}';
  }
}

class RulerPainter extends CustomPainter {
  final int totalMilliseconds;

  RulerPainter({required this.totalMilliseconds});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white24
      ..strokeWidth = 1;
    
    final textPainter = TextPainter(
      textDirection: TextDirection.ltr,
    );
    
    // Draw tick marks every 5 seconds
    final int totalSeconds = (totalMilliseconds / 1000).ceil();
    final double pixelsPerSecond = size.width / (totalMilliseconds / 1000);
    
    for (int i = 0; i <= totalSeconds; i++) {
      final double x = i * pixelsPerSecond;
      
      if (i % 5 == 0) {
        // Major tick
        canvas.drawLine(Offset(x, 0), Offset(x, 12), paint);
        
        // Time label
        final minutes = (i ~/ 60).toString().padLeft(2, '0');
        final seconds = (i % 60).toString().padLeft(2, '0');
        textPainter.text = TextSpan(
          text: '$minutes:$seconds',
          style: const TextStyle(
            color: Colors.white54,
            fontSize: 9,
          ),
        );
        textPainter.layout();
        textPainter.paint(canvas, Offset(x + 2, 10));
      } else {
        // Minor tick
        canvas.drawLine(Offset(x, 0), Offset(x, 6), paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant RulerPainter oldDelegate) => 
      oldDelegate.totalMilliseconds != totalMilliseconds;
}

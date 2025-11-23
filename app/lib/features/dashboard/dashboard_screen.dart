import 'package:flutter/material.dart';
import 'models/dashboard_state.dart';
import 'widgets/editor_app_bar.dart';
import 'widgets/image_viewer.dart';
import 'widgets/player_view.dart';
import 'widgets/timeline_panel.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final DashboardState _dashboardState = DashboardState();

  @override
  void initState() {
    super.initState();
    _dashboardState.initializeVideo('assets/test.MP4');
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: _dashboardState,
      builder: (context, child) {
        return Scaffold(
          backgroundColor: Colors.grey[100], // Light background for the screen
          appBar: const EditorAppBar(),
          body: Column(
            children: [
              // Main Workspace
              Expanded(
                flex: 3,
                child: Row(
                  children: [
                    // Left Column: Player
                    Expanded(
                      flex: 1,
                      child: Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: PlayerView(state: _dashboardState),
                      ),
                    ),
                    
                    // Right Column: Image Viewer
                    const Expanded(
                      flex: 1,
                      child: Padding(
                        padding: EdgeInsets.all(8.0),
                        child: ImageViewer(),
                      ),
                    ),
                  ],
                ),
              ),

              // Timeline
              Expanded(
                flex: 1,
                child: TimelinePanel(state: _dashboardState),
              ),
            ],
          ),
        );
      },
    );
  }

  @override
  void dispose() {
    _dashboardState.dispose();
    super.dispose();
  }
}

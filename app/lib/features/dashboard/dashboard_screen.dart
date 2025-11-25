import 'package:flutter/material.dart';
import 'models/dashboard_state.dart';
import 'widgets/editor_app_bar.dart';
import 'widgets/canvas_video_player.dart';
import 'widgets/documents_panel.dart';
import 'widgets/player_view.dart';
import 'widgets/timeline_panel.dart';

/// Тип верхней панели для мобильной версии
enum TopPanelType { player, canvas }

/// Тип нижней панели для мобильной версии
enum BottomPanelType { documents, timeline }

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final DashboardState _dashboardState = DashboardState();
  
  // Initial height for the timeline
  double _timelineHeight = 300.0;
  
  // Left column width ratio (0.0 to 1.0)
  double _leftColumnRatio = 0.5;
  
  // Overlay state
  OverlayType _selectedOverlay = OverlayType.none;
  
  // Timeline visibility
  bool _isTimelineVisible = true;
  
  // Documents panel width
  double _documentsPanelWidth = 320.0;

  // Mobile panel selection
  TopPanelType _selectedTopPanel = TopPanelType.player;
  BottomPanelType _selectedBottomPanel = BottomPanelType.timeline;

  @override
  void initState() {
    super.initState();
    _dashboardState.initializeVideo('assets/test.MP4');
  }

  /// Проверяет, нужно ли использовать мобильную верстку
  /// Мобильная верстка активируется когда высота > ширины И ширина < 900
  bool _isMobileLayout(BoxConstraints constraints) {
    final size = constraints.biggest;
    return size.height > size.width && size.width < 900;
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: _dashboardState,
      builder: (context, child) {
        return Scaffold(
          backgroundColor: const Color(0xFF1E1E1E),
          appBar: EditorAppBar(
            selectedOverlay: _selectedOverlay,
            onOverlayChanged: (overlay) {
              setState(() {
                _selectedOverlay = overlay;
              });
            },
            dashboardState: _dashboardState,
          ),
          body: LayoutBuilder(
            builder: (context, constraints) {
              if (_isMobileLayout(constraints)) {
                return _buildMobileLayout();
              }
              return _buildDesktopLayout(context);
            },
          ),
        );
      },
    );
  }

  /// Мобильная верстка с двумя панелями и переключателями
  Widget _buildMobileLayout() {
    return Column(
      children: [
        // Верхняя секция с переключателем
        _buildMobileSectionHeader(
          tabs: [
            _MobileTab(
              label: 'Плеер',
              icon: Icons.play_circle_outline,
              isSelected: _selectedTopPanel == TopPanelType.player,
              onTap: () => setState(() => _selectedTopPanel = TopPanelType.player),
            ),
            _MobileTab(
              label: 'Канва',
              icon: Icons.grid_on,
              isSelected: _selectedTopPanel == TopPanelType.canvas,
              onTap: () => setState(() => _selectedTopPanel = TopPanelType.canvas),
            ),
          ],
        ),
        
        // Верхняя панель
        Expanded(
          flex: 3,
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 200),
            child: _selectedTopPanel == TopPanelType.player
                ? Padding(
                    key: const ValueKey('player'),
                    padding: const EdgeInsets.all(8.0),
                    child: PlayerView(state: _dashboardState),
                  )
                : CanvasVideoPlayer(
                    key: const ValueKey('canvas'),
                    state: _dashboardState,
                    overlayType: _selectedOverlay,
                    onCloseOverlay: () {
                      setState(() {
                        _selectedOverlay = OverlayType.none;
                      });
                    },
                  ),
          ),
        ),
        
        // Разделитель
        Container(
          height: 2,
          color: const Color(0xFF3C3C3C),
        ),
        
        // Нижняя секция с переключателем
        _buildMobileSectionHeader(
          tabs: [
            _MobileTab(
              label: 'Документы',
              icon: Icons.folder_outlined,
              isSelected: _selectedBottomPanel == BottomPanelType.documents,
              onTap: () => setState(() => _selectedBottomPanel = BottomPanelType.documents),
            ),
            _MobileTab(
              label: 'Таймлайн',
              icon: Icons.timeline,
              isSelected: _selectedBottomPanel == BottomPanelType.timeline,
              onTap: () => setState(() => _selectedBottomPanel = BottomPanelType.timeline),
            ),
          ],
        ),
        
        // Нижняя панель
        Expanded(
          flex: 2,
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 200),
            child: _selectedBottomPanel == BottomPanelType.documents
                ? const DocumentsPanel(key: ValueKey('documents'))
                : TimelinePanel(key: const ValueKey('timeline'), state: _dashboardState),
          ),
        ),
      ],
    );
  }

  /// Заголовок секции с табами для мобильной версии
  Widget _buildMobileSectionHeader({required List<_MobileTab> tabs}) {
    return Container(
      height: 44,
      color: const Color(0xFF2D2D30),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      child: Row(
        children: tabs.map((tab) {
          return Expanded(
            child: GestureDetector(
              onTap: tab.onTap,
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 150),
                margin: const EdgeInsets.symmetric(horizontal: 4),
                decoration: BoxDecoration(
                  color: tab.isSelected
                      ? const Color(0xFF4CAF50).withOpacity(0.2)
                      : const Color(0xFF3C3C3C),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: tab.isSelected
                        ? const Color(0xFF4CAF50)
                        : const Color(0xFF5A5A5A),
                    width: tab.isSelected ? 1.5 : 1,
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      tab.icon,
                      size: 16,
                      color: tab.isSelected
                          ? const Color(0xFF4CAF50)
                          : const Color(0xFFAAAAAA),
                    ),
                    const SizedBox(width: 6),
                    Text(
                      tab.label,
                      style: TextStyle(
                        color: tab.isSelected
                            ? const Color(0xFF4CAF50)
                            : const Color(0xFFAAAAAA),
                        fontSize: 13,
                        fontWeight: tab.isSelected ? FontWeight.w600 : FontWeight.w400,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  /// Десктопная верстка (оригинальная)
  Widget _buildDesktopLayout(BuildContext context) {
    return Column(
      children: [
        // Main Workspace
        Expanded(
          child: LayoutBuilder(
            builder: (context, constraints) {
              final double totalWidth = constraints.maxWidth;
              final double leftWidth = totalWidth * _leftColumnRatio;
              final double rightWidth = totalWidth - leftWidth - 8; // -8 for divider

              return Row(
                children: [
                  // Left Column: Player
                  SizedBox(
                    width: leftWidth,
                    child: Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: PlayerView(state: _dashboardState),
                    ),
                  ),
                  
                  // Draggable Vertical Divider
                  GestureDetector(
                    onHorizontalDragUpdate: (details) {
                      setState(() {
                        // Update ratio based on drag
                        double newRatio = _leftColumnRatio + (details.delta.dx / totalWidth);
                        // Clamp ratio (min 20%, max 80%)
                        _leftColumnRatio = newRatio.clamp(0.2, 0.8);
                      });
                    },
                    child: MouseRegion(
                      cursor: SystemMouseCursors.resizeColumn,
                      child: Container(
                        width: 8,
                        color: const Color(0xFF3C3C3C),
                        child: Center(
                          child: Container(
                            width: 4,
                            height: 40,
                            decoration: BoxDecoration(
                              color: const Color(0xFF5A5A5A),
                              borderRadius: BorderRadius.circular(2),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                  
                  // Right Column: Canvas Video Player with Overlay
                  SizedBox(
                    width: rightWidth,
                    child: CanvasVideoPlayer(
                      state: _dashboardState,
                      overlayType: _selectedOverlay,
                      onCloseOverlay: () {
                        setState(() {
                          _selectedOverlay = OverlayType.none;
                        });
                      },
                    ),
                  ),
                ],
              );
            }
          ),
        ),

        // Draggable Horizontal Divider with Toggle Button
        Container(
          height: 24,
          color: const Color(0xFF3C3C3C),
          child: Row(
            children: [
              // Left spacer with drag area
              Expanded(
                child: GestureDetector(
                  onVerticalDragUpdate: _isTimelineVisible
                      ? (details) {
                          setState(() {
                            _timelineHeight -= details.delta.dy;
                            final double totalHeight = MediaQuery.of(context).size.height;
                            _timelineHeight = _timelineHeight.clamp(100.0, totalHeight - 150.0);
                          });
                        }
                      : null,
                  child: MouseRegion(
                    cursor: _isTimelineVisible 
                        ? SystemMouseCursors.resizeRow 
                        : SystemMouseCursors.basic,
                    child: Center(
                      child: _isTimelineVisible
                          ? Container(
                              width: 40,
                              height: 4,
                              decoration: BoxDecoration(
                                color: const Color(0xFF5A5A5A),
                                borderRadius: BorderRadius.circular(2),
                              ),
                            )
                          : null,
                    ),
                  ),
                ),
              ),
              // Toggle button
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                child: InkWell(
                  onTap: () {
                    setState(() {
                      _isTimelineVisible = !_isTimelineVisible;
                    });
                  },
                  borderRadius: BorderRadius.circular(4),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: const Color(0xFF2D2D30),
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(
                        color: const Color(0xFF5A5A5A),
                        width: 1,
                      ),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          _isTimelineVisible 
                              ? Icons.keyboard_arrow_down 
                              : Icons.keyboard_arrow_up,
                          size: 16,
                          color: const Color(0xFFCCCCCC),
                        ),
                        const SizedBox(width: 4),
                        Text(
                          _isTimelineVisible ? 'Скрыть' : 'Показать',
                          style: const TextStyle(
                            color: Color(0xFFCCCCCC),
                            fontSize: 11,
                            fontWeight: FontWeight.w500,
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

        // Bottom Panel: Documents + Timeline (Resizable)
        if (_isTimelineVisible)
          SizedBox(
            height: _timelineHeight,
            child: Row(
              children: [
                // Documents Panel
                SizedBox(
                  width: _documentsPanelWidth,
                  child: const DocumentsPanel(),
                ),
                
                // Draggable Vertical Divider between documents and timeline
                GestureDetector(
                  onHorizontalDragUpdate: (details) {
                    setState(() {
                      _documentsPanelWidth += details.delta.dx;
                      _documentsPanelWidth = _documentsPanelWidth.clamp(200.0, 500.0);
                    });
                  },
                  child: MouseRegion(
                    cursor: SystemMouseCursors.resizeColumn,
                    child: Container(
                      width: 6,
                      color: const Color(0xFF3C3C3C),
                      child: Center(
                        child: Container(
                          width: 3,
                          height: 30,
                          decoration: BoxDecoration(
                            color: const Color(0xFF5A5A5A),
                            borderRadius: BorderRadius.circular(2),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
                
                // Timeline Panel
                Expanded(
                  child: TimelinePanel(state: _dashboardState),
                ),
              ],
            ),
          ),
      ],
    );
  }

  @override
  void dispose() {
    _dashboardState.dispose();
    super.dispose();
  }
}

/// Вспомогательный класс для мобильного таба
class _MobileTab {
  final String label;
  final IconData icon;
  final bool isSelected;
  final VoidCallback onTap;

  _MobileTab({
    required this.label,
    required this.icon,
    required this.isSelected,
    required this.onTap,
  });
}

import 'package:flutter/material.dart';
import '../models/dashboard_state.dart';
import '../services/pdf_export_service.dart';

enum OverlayType { none, light }

class EditorAppBar extends StatefulWidget implements PreferredSizeWidget {
  final OverlayType selectedOverlay;
  final ValueChanged<OverlayType> onOverlayChanged;
  final DashboardState dashboardState;

  const EditorAppBar({
    super.key,
    required this.selectedOverlay,
    required this.onOverlayChanged,
    required this.dashboardState,
  });

  @override
  State<EditorAppBar> createState() => _EditorAppBarState();

  @override
  Size get preferredSize => const Size.fromHeight(32);
}

class _EditorAppBarState extends State<EditorAppBar> {
  bool _isMenuHovered = false;
  OverlayEntry? _tooltipOverlay;

  final List<_MenuItemData> _menuItems = [
    _MenuItemData('Файл', Icons.folder_outlined),
    _MenuItemData('Правка', Icons.edit_outlined),
    _MenuItemData('Вид', Icons.visibility_outlined),
    _MenuItemData('Вставка', Icons.add_box_outlined),
    _MenuItemData('Эффекты', Icons.auto_fix_high_outlined),
    _MenuItemData('Справка', Icons.help_outline),
  ];

  void _showTooltip(BuildContext context, Offset position) {
    _hideTooltip();
    
    _tooltipOverlay = OverlayEntry(
      builder: (context) => Positioned(
        left: position.dx + 12,
        top: position.dy + 12,
        child: Material(
          color: Colors.transparent,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: const Color(0xFF1E1E1E),
              borderRadius: BorderRadius.circular(4),
              border: Border.all(color: const Color(0xFF5A5A5A)),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.3),
                  blurRadius: 8,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: const Text(
              'Ещё в разработке',
              style: TextStyle(
                color: Colors.white70,
                fontSize: 12,
              ),
            ),
          ),
        ),
      ),
    );
    
    Overlay.of(context).insert(_tooltipOverlay!);
  }

  void _hideTooltip() {
    _tooltipOverlay?.remove();
    _tooltipOverlay = null;
  }

  void _showDevelopmentDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => const _DevelopmentDialog(),
    );
  }

  void _showExportDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => _ExportDialog(
        groups: widget.dashboardState.timelineGroups,
      ),
    );
  }

  @override
  void dispose() {
    _hideTooltip();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AppBar(
      backgroundColor: const Color(0xFF2D2D30),
      elevation: 0,
      automaticallyImplyLeading: false,
      titleSpacing: 0,
      toolbarHeight: 32,
      title: Row(
        children: [
          // Menu Bar (File, Edit, View, etc.)
          MouseRegion(
            onEnter: (event) {
              setState(() => _isMenuHovered = true);
              _showTooltip(context, event.position);
            },
            onHover: (event) {
              _hideTooltip();
              _showTooltip(context, event.position);
            },
            onExit: (_) {
              setState(() => _isMenuHovered = false);
              _hideTooltip();
            },
            child: GestureDetector(
              onTap: () => _showDevelopmentDialog(context),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 150),
                padding: const EdgeInsets.symmetric(horizontal: 8),
                decoration: BoxDecoration(
                  color: _isMenuHovered 
                      ? const Color(0xFF094771) 
                      : Colors.transparent,
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: _menuItems.map((item) => _buildMenuButton(item)).toList(),
                ),
              ),
            ),
          ),
          
          const Spacer(),
        ],
      ),
      actions: [
        // Overlay Selector
        _buildOverlaySelector(context),
        const SizedBox(width: 8),
        
        // Export Button
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0),
          child: ElevatedButton(
            onPressed: () => _showExportDialog(context),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF4CAF50),
              foregroundColor: Colors.white,
              elevation: 0,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 0),
              minimumSize: const Size(0, 24),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(4),
              ),
            ),
            child: const Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.file_download_outlined, size: 14),
                SizedBox(width: 4),
                Text('Экспорт', style: TextStyle(fontSize: 12)),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildMenuButton(_MenuItemData item) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      child: Text(
        item.label,
        style: const TextStyle(
          color: Colors.white70,
          fontSize: 12,
          fontWeight: FontWeight.w400,
        ),
      ),
    );
  }

  Widget _buildOverlaySelector(BuildContext context) {
    return PopupMenuButton<OverlayType>(
      offset: const Offset(0, 32),
      color: const Color(0xFF2D2D30),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(4),
        side: const BorderSide(color: Color(0xFF3C3C3C)),
      ),
      tooltip: 'Слои',
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
        decoration: BoxDecoration(
          color: widget.selectedOverlay != OverlayType.none
              ? const Color(0xFF4CAF50).withOpacity(0.2)
              : const Color(0xFF3C3C3C),
          borderRadius: BorderRadius.circular(4),
          border: Border.all(
            color: widget.selectedOverlay != OverlayType.none
                ? const Color(0xFF4CAF50)
                : const Color(0xFF5A5A5A),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.layers_outlined,
              size: 14,
              color: widget.selectedOverlay != OverlayType.none
                  ? const Color(0xFF4CAF50)
                  : Colors.white70,
            ),
            const SizedBox(width: 4),
            Text(
              'Слои',
              style: TextStyle(
                color: widget.selectedOverlay != OverlayType.none
                    ? const Color(0xFF4CAF50)
                    : Colors.white70,
                fontSize: 12,
              ),
            ),
            const SizedBox(width: 2),
            Icon(
              Icons.arrow_drop_down,
              size: 14,
              color: widget.selectedOverlay != OverlayType.none
                  ? const Color(0xFF4CAF50)
                  : Colors.white70,
            ),
          ],
        ),
      ),
      itemBuilder: (context) => [
        _buildMenuItem(
          OverlayType.light,
          'Свет',
          Icons.lightbulb_outline,
          const Color(0xFFFFEB3B),
        ),
      ],
      onSelected: (value) {
        if (widget.selectedOverlay == value) {
          widget.onOverlayChanged(OverlayType.none);
        } else {
          widget.onOverlayChanged(value);
        }
      },
    );
  }

  PopupMenuItem<OverlayType> _buildMenuItem(
    OverlayType type,
    String label,
    IconData icon,
    Color accentColor,
  ) {
    final bool isSelected = widget.selectedOverlay == type;
    
    return PopupMenuItem<OverlayType>(
      value: type,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          children: [
            Container(
              width: 24,
              height: 24,
              decoration: BoxDecoration(
                color: isSelected ? accentColor.withOpacity(0.2) : Colors.transparent,
                borderRadius: BorderRadius.circular(4),
              ),
              child: Icon(
                icon,
                size: 16,
                color: isSelected ? accentColor : Colors.white70,
              ),
            ),
            const SizedBox(width: 12),
            Text(
              label,
              style: TextStyle(
                color: isSelected ? accentColor : Colors.white,
                fontSize: 13,
              ),
            ),
            const Spacer(),
            if (isSelected)
              Icon(
                Icons.check,
                size: 16,
                color: accentColor,
              ),
          ],
        ),
      ),
    );
  }
}

class _MenuItemData {
  final String label;
  final IconData icon;

  _MenuItemData(this.label, this.icon);
}

class _DevelopmentDialog extends StatelessWidget {
  const _DevelopmentDialog();

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Colors.transparent,
      child: Container(
        width: 400,
        decoration: BoxDecoration(
          color: const Color(0xFFF0F0F0),
          borderRadius: BorderRadius.circular(8),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.3),
              blurRadius: 20,
              offset: const Offset(0, 10),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Title Bar (Windows-style)
            Container(
              height: 32,
              decoration: const BoxDecoration(
                color: Color(0xFF2D2D30),
                borderRadius: BorderRadius.vertical(top: Radius.circular(8)),
              ),
              child: Row(
                children: [
                  const SizedBox(width: 12),
                  const Icon(
                    Icons.info_outline,
                    size: 16,
                    color: Colors.white70,
                  ),
                  const SizedBox(width: 8),
                  const Text(
                    'StageFlow Editor',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                    ),
                  ),
                  const Spacer(),
                  // Close button
                  InkWell(
                    onTap: () => Navigator.of(context).pop(),
                    child: Container(
                      width: 46,
                      height: 32,
                      decoration: const BoxDecoration(
                        borderRadius: BorderRadius.only(topRight: Radius.circular(8)),
                      ),
                      child: const Center(
                        child: Icon(
                          Icons.close,
                          size: 16,
                          color: Colors.white70,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            
            // Content
            Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                children: [
                  Container(
                    width: 64,
                    height: 64,
                    decoration: BoxDecoration(
                      color: const Color(0xFF094771),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(
                      Icons.construction,
                      size: 32,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'Функция в разработке',
                    style: TextStyle(
                      color: Color(0xFF1E1E1E),
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Данный раздел меню находится в активной разработке и будет доступен в следующих версиях приложения.',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Color(0xFF5A5A5A),
                      fontSize: 13,
                    ),
                  ),
                  const SizedBox(height: 24),
                  
                  // OK Button
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: () => Navigator.of(context).pop(),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF094771),
                        foregroundColor: Colors.white,
                        elevation: 0,
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(4),
                        ),
                      ),
                      child: const Text('OK'),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ExportDialog extends StatefulWidget {
  final List<TimelineGroup> groups;

  const _ExportDialog({required this.groups});

  @override
  State<_ExportDialog> createState() => _ExportDialogState();
}

class _ExportDialogState extends State<_ExportDialog> {
  final Set<String> _selectedGroups = {};
  final TextEditingController _titleController = TextEditingController(text: 'Спектакль');
  bool _isExporting = false;

  @override
  void initState() {
    super.initState();
    // Select all groups by default
    for (final group in widget.groups) {
      _selectedGroups.add(group.name);
    }
  }

  @override
  void dispose() {
    _titleController.dispose();
    super.dispose();
  }

  Future<void> _export() async {
    if (_selectedGroups.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Выберите хотя бы одну группу для экспорта'),
          backgroundColor: Color(0xFFE53935),
        ),
      );
      return;
    }

    setState(() => _isExporting = true);

    try {
      await PdfExportService.exportToPdf(
        groups: widget.groups,
        selectedGroupNames: _selectedGroups,
        showTitle: _titleController.text.isEmpty ? 'Спектакль' : _titleController.text,
      );
      
      if (mounted) {
        Navigator.of(context).pop();
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Ошибка экспорта: $e'),
            backgroundColor: const Color(0xFFE53935),
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isExporting = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Colors.transparent,
      child: Container(
        width: 480,
        decoration: BoxDecoration(
          color: const Color(0xFF2D2D30),
          borderRadius: BorderRadius.circular(8),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.4),
              blurRadius: 24,
              offset: const Offset(0, 12),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Title Bar
            Container(
              height: 36,
              padding: const EdgeInsets.symmetric(horizontal: 12),
              decoration: const BoxDecoration(
                color: Color(0xFF1E1E1E),
                borderRadius: BorderRadius.vertical(top: Radius.circular(8)),
              ),
              child: Row(
                children: [
                  const Icon(
                    Icons.picture_as_pdf,
                    size: 16,
                    color: Color(0xFFE53935),
                  ),
                  const SizedBox(width: 8),
                  const Text(
                    'Экспорт планшета спектакля',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const Spacer(),
                  InkWell(
                    onTap: () => Navigator.of(context).pop(),
                    borderRadius: BorderRadius.circular(4),
                    child: Container(
                      width: 28,
                      height: 28,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: const Center(
                        child: Icon(
                          Icons.close,
                          size: 16,
                          color: Colors.white54,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            
            // Content
            Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Show Title Input
                  const Text(
                    'Название спектакля',
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 12,
                    ),
                  ),
                  const SizedBox(height: 8),
                  TextField(
                    controller: _titleController,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                    ),
                    decoration: InputDecoration(
                      hintText: 'Введите название...',
                      hintStyle: const TextStyle(color: Colors.white30),
                      filled: true,
                      fillColor: const Color(0xFF1E1E1E),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(4),
                        borderSide: const BorderSide(color: Color(0xFF3C3C3C)),
                      ),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(4),
                        borderSide: const BorderSide(color: Color(0xFF3C3C3C)),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(4),
                        borderSide: const BorderSide(color: Color(0xFF4CAF50)),
                      ),
                    ),
                  ),
                  
                  const SizedBox(height: 20),
                  
                  // Groups Selection
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        'Выберите группы для экспорта',
                        style: TextStyle(
                          color: Colors.white70,
                          fontSize: 12,
                        ),
                      ),
                      Row(
                        children: [
                          TextButton(
                            onPressed: () {
                              setState(() {
                                _selectedGroups.clear();
                                for (final group in widget.groups) {
                                  _selectedGroups.add(group.name);
                                }
                              });
                            },
                            style: TextButton.styleFrom(
                              foregroundColor: const Color(0xFF4CAF50),
                              padding: const EdgeInsets.symmetric(horizontal: 8),
                              minimumSize: Size.zero,
                              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                            ),
                            child: const Text('Все', style: TextStyle(fontSize: 11)),
                          ),
                          const Text(' / ', style: TextStyle(color: Colors.white30, fontSize: 11)),
                          TextButton(
                            onPressed: () {
                              setState(() => _selectedGroups.clear());
                            },
                            style: TextButton.styleFrom(
                              foregroundColor: Colors.white54,
                              padding: const EdgeInsets.symmetric(horizontal: 8),
                              minimumSize: Size.zero,
                              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                            ),
                            child: const Text('Ничего', style: TextStyle(fontSize: 11)),
                          ),
                        ],
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  
                  // Groups List
                  Container(
                    constraints: const BoxConstraints(maxHeight: 250),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1E1E1E),
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(color: const Color(0xFF3C3C3C)),
                    ),
                    child: ListView.builder(
                      shrinkWrap: true,
                      itemCount: widget.groups.length,
                      itemBuilder: (context, index) {
                        final group = widget.groups[index];
                        final isSelected = _selectedGroups.contains(group.name);
                        final elementsCount = group.elements.length + 
                            group.subGroups.fold<int>(0, (sum, sg) => sum + sg.elements.length);
                        
                        return InkWell(
                          onTap: () {
                            setState(() {
                              if (isSelected) {
                                _selectedGroups.remove(group.name);
                              } else {
                                _selectedGroups.add(group.name);
                              }
                            });
                          },
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                            decoration: BoxDecoration(
                              border: index < widget.groups.length - 1
                                  ? const Border(bottom: BorderSide(color: Color(0xFF3C3C3C), width: 0.5))
                                  : null,
                            ),
                            child: Row(
                              children: [
                                Container(
                                  width: 20,
                                  height: 20,
                                  decoration: BoxDecoration(
                                    color: isSelected ? const Color(0xFF4CAF50) : Colors.transparent,
                                    borderRadius: BorderRadius.circular(4),
                                    border: Border.all(
                                      color: isSelected ? const Color(0xFF4CAF50) : const Color(0xFF5A5A5A),
                                    ),
                                  ),
                                  child: isSelected
                                      ? const Icon(Icons.check, size: 14, color: Colors.white)
                                      : null,
                                ),
                                const SizedBox(width: 12),
                                Container(
                                  width: 4,
                                  height: 24,
                                  decoration: BoxDecoration(
                                    color: group.color,
                                    borderRadius: BorderRadius.circular(2),
                                  ),
                                ),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        group.name,
                                        style: TextStyle(
                                          color: isSelected ? Colors.white : Colors.white54,
                                          fontSize: 13,
                                          fontWeight: FontWeight.w500,
                                        ),
                                      ),
                                      Text(
                                        '$elementsCount элемент${_getElementsEnding(elementsCount)}',
                                        style: const TextStyle(
                                          color: Colors.white30,
                                          fontSize: 11,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                  
                  const SizedBox(height: 20),
                  
                  // Actions
                  Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      TextButton(
                        onPressed: () => Navigator.of(context).pop(),
                        style: TextButton.styleFrom(
                          foregroundColor: Colors.white54,
                          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                        ),
                        child: const Text('Отмена'),
                      ),
                      const SizedBox(width: 12),
                      ElevatedButton(
                        onPressed: _isExporting ? null : _export,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF4CAF50),
                          foregroundColor: Colors.white,
                          disabledBackgroundColor: const Color(0xFF4CAF50).withOpacity(0.5),
                          elevation: 0,
                          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 10),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(4),
                          ),
                        ),
                        child: _isExporting
                            ? const SizedBox(
                                width: 16,
                                height: 16,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white,
                                ),
                              )
                            : const Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(Icons.picture_as_pdf, size: 16),
                                  SizedBox(width: 8),
                                  Text('Экспорт в PDF'),
                                ],
                              ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _getElementsEnding(int count) {
    if (count % 10 == 1 && count % 100 != 11) return '';
    if (count % 10 >= 2 && count % 10 <= 4 && (count % 100 < 10 || count % 100 >= 20)) return 'а';
    return 'ов';
  }
}

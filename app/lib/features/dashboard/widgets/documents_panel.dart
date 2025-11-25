import 'package:flutter/material.dart';

class DocumentItem {
  final String name;
  final String date;
  final String size;
  final IconData icon;
  final Color color;

  const DocumentItem({
    required this.name,
    required this.date,
    required this.size,
    this.icon = Icons.picture_as_pdf,
    this.color = const Color(0xFFE53935),
  });
}

class DocumentsPanel extends StatefulWidget {
  final VoidCallback? onDocumentSelected;

  const DocumentsPanel({super.key, this.onDocumentSelected});

  @override
  State<DocumentsPanel> createState() => _DocumentsPanelState();
}

class _DocumentsPanelState extends State<DocumentsPanel> {
  int? _selectedIndex;
  int? _hoveredIndex;

  final List<DocumentItem> _documents = const [
    DocumentItem(
      name: 'Гримерный цех.pdf',
      date: '22.02.2023',
      size: '14,3 MB',
      color: Color(0xFFE91E63),
    ),
    DocumentItem(
      name: 'Звуковой цех.pdf',
      date: '22.02.2023',
      size: '1,8 MB',
      color: Color(0xFF2196F3),
    ),
    DocumentItem(
      name: 'Костюмерный цех.pdf',
      date: '22.02.2023',
      size: '1,6 MB',
      color: Color(0xFF9C27B0),
    ),
    DocumentItem(
      name: 'Машино-декорационный цех.pdf',
      date: '22.02.2023',
      size: '1 MB',
      color: Color(0xFF795548),
    ),
    DocumentItem(
      name: 'Реквизиторский цех.pdf',
      date: '22.02.2023',
      size: '16,2 MB',
      color: Color(0xFF4CAF50),
    ),
    DocumentItem(
      name: 'Хореография.pdf',
      date: '22.02.2023',
      size: '15 MB',
      color: Color(0xFFFF9800),
    ),
    DocumentItem(
      name: 'Свет.pdf',
      date: '22.02.2023',
      size: '700 КБ',
      color: Color(0xFFFFEB3B),
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Color(0xFF252526),
        border: Border(
          right: BorderSide(color: Color(0xFF3C3C3C), width: 1),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Header
          Container(
            height: 40,
            padding: const EdgeInsets.symmetric(horizontal: 12),
            decoration: const BoxDecoration(
              color: Color(0xFF3C3C3C),
              border: Border(
                bottom: BorderSide(color: Color(0xFF1E1E1E), width: 1),
              ),
            ),
            child: Row(
              children: [
                const Icon(
                  Icons.folder_outlined,
                  color: Color(0xFFCCCCCC),
                  size: 18,
                ),
                const SizedBox(width: 8),
                const Text(
                  'Документы',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.3,
                  ),
                ),
                const Spacer(),
                // Documents count badge
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: const Color(0xFF007ACC),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    '${_documents.length}',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
          ),

          // Column Headers
          Container(
            height: 28,
            padding: const EdgeInsets.symmetric(horizontal: 12),
            decoration: const BoxDecoration(
              color: Color(0xFF2D2D30),
              border: Border(
                bottom: BorderSide(color: Color(0xFF3C3C3C), width: 1),
              ),
            ),
            child: const Row(
              children: [
                SizedBox(width: 28), // Icon space
                Expanded(
                  flex: 5,
                  child: Text(
                    'Название',
                    style: TextStyle(
                      color: Color(0xFF808080),
                      fontSize: 10,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
                Expanded(
                  flex: 2,
                  child: Text(
                    'Дата',
                    style: TextStyle(
                      color: Color(0xFF808080),
                      fontSize: 10,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
                SizedBox(
                  width: 60,
                  child: Text(
                    'Размер',
                    textAlign: TextAlign.right,
                    style: TextStyle(
                      color: Color(0xFF808080),
                      fontSize: 10,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ],
            ),
          ),

          // Documents List
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(vertical: 4),
              itemCount: _documents.length,
              itemBuilder: (context, index) {
                final doc = _documents[index];
                final isSelected = _selectedIndex == index;
                final isHovered = _hoveredIndex == index;

                return MouseRegion(
                  onEnter: (_) => setState(() => _hoveredIndex = index),
                  onExit: (_) => setState(() => _hoveredIndex = null),
                  child: GestureDetector(
                    onTap: () {
                      setState(() => _selectedIndex = index);
                      widget.onDocumentSelected?.call();
                    },
                    onDoubleTap: () {
                      // TODO: Open document
                      debugPrint('Opening: ${doc.name}');
                    },
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 150),
                      height: 36,
                      margin: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                      padding: const EdgeInsets.symmetric(horizontal: 8),
                      decoration: BoxDecoration(
                        color: isSelected
                            ? const Color(0xFF094771)
                            : isHovered
                                ? const Color(0xFF2A2D2E)
                                : Colors.transparent,
                        borderRadius: BorderRadius.circular(4),
                        border: isSelected
                            ? Border.all(color: const Color(0xFF007ACC), width: 1)
                            : null,
                      ),
                      child: Row(
                        children: [
                          // PDF Icon
                          Container(
                            width: 24,
                            height: 24,
                            decoration: BoxDecoration(
                              color: doc.color.withOpacity(0.15),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Icon(
                              Icons.picture_as_pdf_rounded,
                              size: 16,
                              color: doc.color,
                            ),
                          ),
                          const SizedBox(width: 8),
                          // Name
                          Expanded(
                            flex: 5,
                            child: Text(
                              doc.name,
                              style: TextStyle(
                                color: isSelected ? Colors.white : const Color(0xFFCCCCCC),
                                fontSize: 12,
                                fontWeight: isSelected ? FontWeight.w500 : FontWeight.normal,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          // Date
                          Expanded(
                            flex: 2,
                            child: Text(
                              doc.date,
                              style: TextStyle(
                                color: isSelected
                                    ? const Color(0xFFB0B0B0)
                                    : const Color(0xFF808080),
                                fontSize: 11,
                              ),
                            ),
                          ),
                          // Size
                          SizedBox(
                            width: 60,
                            child: Text(
                              doc.size,
                              textAlign: TextAlign.right,
                              style: TextStyle(
                                color: isSelected
                                    ? const Color(0xFFB0B0B0)
                                    : const Color(0xFF808080),
                                fontSize: 11,
                                fontFamily: 'monospace',
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
          ),

          // Footer with info
          Container(
            height: 32,
            padding: const EdgeInsets.symmetric(horizontal: 12),
            decoration: const BoxDecoration(
              color: Color(0xFF2D2D30),
              border: Border(
                top: BorderSide(color: Color(0xFF3C3C3C), width: 1),
              ),
            ),
            child: Row(
              children: [
                Icon(
                  Icons.info_outline,
                  size: 14,
                  color: const Color(0xFF808080),
                ),
                const SizedBox(width: 6),
                Text(
                  _selectedIndex != null
                      ? 'Выбрано: ${_documents[_selectedIndex!].name}'
                      : 'Выберите документ',
                  style: const TextStyle(
                    color: Color(0xFF808080),
                    fontSize: 11,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}


import 'package:flutter/material.dart';

class EditorAppBar extends StatelessWidget implements PreferredSizeWidget {
  const EditorAppBar({super.key});

  @override
  Widget build(BuildContext context) {
    return AppBar(
      backgroundColor: Colors.white, // Light background
      elevation: 1, // Slight shadow for separation
      iconTheme: const IconThemeData(color: Colors.black87), // Dark icons
      leading: IconButton(
        icon: const Icon(Icons.arrow_back),
        onPressed: () {
          // Handle back
        },
      ),
      title: const Text(
        'HackEdit',
        style: TextStyle(color: Colors.black87), // Dark text
      ),
      centerTitle: true,
      actions: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
          child: ElevatedButton(
            onPressed: () {
              // Handle export
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.blueAccent,
              foregroundColor: Colors.white,
              elevation: 0,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(20),
              ),
            ),
            child: const Text('Экспорт'),
          ),
        ),
      ],
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}

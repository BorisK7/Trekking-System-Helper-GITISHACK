import 'package:flutter/material.dart';

class ImageViewer extends StatelessWidget {
  const ImageViewer({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color(0xFFF5F5F5), // Updated background color
      child: ClipRect(
        child: InteractiveViewer(
          minScale: 0.1,
          maxScale: 5.0,
          boundaryMargin: const EdgeInsets.all(double.infinity),
          child: Image.asset(
            'assets/map.png',
            fit: BoxFit.contain,
            errorBuilder: (context, error, stackTrace) {
              return Center(
                child: Text(
                  'Error loading image: $error',
                  style: const TextStyle(color: Colors.red),
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}

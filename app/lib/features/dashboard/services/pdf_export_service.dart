import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';
import '../models/dashboard_state.dart';

class PdfExportService {
  static String _formatDuration(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, "0");
    String twoDigitMinutes = twoDigits(duration.inMinutes.remainder(60));
    String twoDigitSeconds = twoDigits(duration.inSeconds.remainder(60));
    return "$twoDigitMinutes:$twoDigitSeconds";
  }

  static Future<void> exportToPdf({
    required List<TimelineGroup> groups,
    required Set<String> selectedGroupNames,
    required String showTitle,
  }) async {
    // Load font with Cyrillic support
    final font = await PdfGoogleFonts.notoSansRegular();
    final fontBold = await PdfGoogleFonts.notoSansBold();
    final fontItalic = await PdfGoogleFonts.notoSansItalic();

    final pdf = pw.Document();
    
    final selectedGroups = groups
        .where((g) => selectedGroupNames.contains(g.name))
        .toList();

    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        margin: const pw.EdgeInsets.all(40),
        theme: pw.ThemeData.withFont(
          base: font,
          bold: fontBold,
          italic: fontItalic,
        ),
        header: (context) => _buildHeader(showTitle, context, fontBold),
        footer: (context) => _buildFooter(context, font),
        build: (context) => [
          pw.SizedBox(height: 20),
          ...selectedGroups.expand((group) => _buildGroupSection(group, font, fontBold, fontItalic)),
        ],
      ),
    );

    await Printing.layoutPdf(
      onLayout: (format) async => pdf.save(),
      name: 'Planshet_${showTitle.replaceAll(' ', '_')}.pdf',
    );
  }

  static pw.Widget _buildHeader(String showTitle, pw.Context context, pw.Font fontBold) {
    return pw.Container(
      padding: const pw.EdgeInsets.only(bottom: 10),
      decoration: const pw.BoxDecoration(
        border: pw.Border(
          bottom: pw.BorderSide(color: PdfColors.grey400, width: 1),
        ),
      ),
      child: pw.Row(
        mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
        children: [
          pw.Column(
            crossAxisAlignment: pw.CrossAxisAlignment.start,
            children: [
              pw.Text(
                'ПЛАНШЕТ СПЕКТАКЛЯ',
                style: pw.TextStyle(
                  font: fontBold,
                  fontSize: 18,
                  fontWeight: pw.FontWeight.bold,
                  color: PdfColors.grey800,
                ),
              ),
              pw.SizedBox(height: 4),
              pw.Text(
                showTitle,
                style: pw.TextStyle(
                  font: fontBold,
                  fontSize: 14,
                  fontWeight: pw.FontWeight.bold,
                ),
              ),
            ],
          ),
          pw.Text(
            'StageFlow Editor',
            style: const pw.TextStyle(
              fontSize: 10,
              color: PdfColors.grey600,
            ),
          ),
        ],
      ),
    );
  }

  static pw.Widget _buildFooter(pw.Context context, pw.Font font) {
    final now = DateTime.now();
    return pw.Container(
      padding: const pw.EdgeInsets.only(top: 10),
      decoration: const pw.BoxDecoration(
        border: pw.Border(
          top: pw.BorderSide(color: PdfColors.grey300, width: 0.5),
        ),
      ),
      child: pw.Row(
        mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
        children: [
          pw.Text(
            'Сгенерировано: ${now.day.toString().padLeft(2, '0')}.${now.month.toString().padLeft(2, '0')}.${now.year}',
            style: pw.TextStyle(
              font: font,
              fontSize: 9,
              color: PdfColors.grey500,
            ),
          ),
          pw.Text(
            'Страница ${context.pageNumber} из ${context.pagesCount}',
            style: pw.TextStyle(
              font: font,
              fontSize: 9,
              color: PdfColors.grey500,
            ),
          ),
        ],
      ),
    );
  }

  static List<pw.Widget> _buildGroupSection(
    TimelineGroup group,
    pw.Font font,
    pw.Font fontBold,
    pw.Font fontItalic,
  ) {
    final List<pw.Widget> widgets = [];
    
    // Group Header
    widgets.add(
      pw.Container(
        margin: const pw.EdgeInsets.only(top: 16, bottom: 8),
        padding: const pw.EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: pw.BoxDecoration(
          color: _getPdfColor(group.color),
          borderRadius: pw.BorderRadius.circular(4),
        ),
        child: pw.Row(
          children: [
            pw.Text(
              group.name.toUpperCase(),
              style: pw.TextStyle(
                font: fontBold,
                fontSize: 12,
                fontWeight: pw.FontWeight.bold,
                color: PdfColors.white,
              ),
            ),
          ],
        ),
      ),
    );

    // Table Header
    widgets.add(
      pw.Container(
        padding: const pw.EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        decoration: const pw.BoxDecoration(
          color: PdfColors.grey200,
          border: pw.Border(
            bottom: pw.BorderSide(color: PdfColors.grey400, width: 1),
          ),
        ),
        child: pw.Row(
          children: [
            pw.Expanded(
              flex: 3,
              child: pw.Text(
                'Название',
                style: pw.TextStyle(
                  font: fontBold,
                  fontSize: 10,
                  fontWeight: pw.FontWeight.bold,
                ),
              ),
            ),
            pw.Expanded(
              flex: 1,
              child: pw.Text(
                'Начало',
                style: pw.TextStyle(
                  font: fontBold,
                  fontSize: 10,
                  fontWeight: pw.FontWeight.bold,
                ),
                textAlign: pw.TextAlign.center,
              ),
            ),
            pw.Expanded(
              flex: 1,
              child: pw.Text(
                'Конец',
                style: pw.TextStyle(
                  font: fontBold,
                  fontSize: 10,
                  fontWeight: pw.FontWeight.bold,
                ),
                textAlign: pw.TextAlign.center,
              ),
            ),
            pw.Expanded(
              flex: 2,
              child: pw.Text(
                'Примечание',
                style: pw.TextStyle(
                  font: fontBold,
                  fontSize: 10,
                  fontWeight: pw.FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
      ),
    );

    // SubGroups
    for (final subGroup in group.subGroups) {
      // SubGroup Header
      widgets.add(
        pw.Container(
          padding: const pw.EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: const pw.BoxDecoration(
            color: PdfColors.grey100,
          ),
          child: pw.Text(
            subGroup.name,
            style: pw.TextStyle(
              font: fontItalic,
              fontSize: 10,
              fontWeight: pw.FontWeight.bold,
              fontStyle: pw.FontStyle.italic,
            ),
          ),
        ),
      );

      // SubGroup Elements
      for (final element in subGroup.elements) {
        widgets.add(_buildElementRow(element, font, fontBold));
      }
    }

    // Direct Elements
    for (final element in group.elements) {
      widgets.add(_buildElementRow(element, font, fontBold));
    }

    return widgets;
  }

  static pw.Widget _buildElementRow(TimelineElement element, pw.Font font, pw.Font fontBold) {
    return pw.Container(
      padding: const pw.EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      decoration: const pw.BoxDecoration(
        border: pw.Border(
          bottom: pw.BorderSide(color: PdfColors.grey200, width: 0.5),
        ),
      ),
      child: pw.Row(
        children: [
          pw.Expanded(
            flex: 3,
            child: pw.Text(
              element.name,
              style: pw.TextStyle(font: font, fontSize: 10),
            ),
          ),
          pw.Expanded(
            flex: 1,
            child: pw.Text(
              _formatDuration(element.startTime),
              style: pw.TextStyle(
                font: fontBold,
                fontSize: 10,
                fontWeight: pw.FontWeight.bold,
              ),
              textAlign: pw.TextAlign.center,
            ),
          ),
          pw.Expanded(
            flex: 1,
            child: pw.Text(
              _formatDuration(element.endTime),
              style: pw.TextStyle(
                font: fontBold,
                fontSize: 10,
                fontWeight: pw.FontWeight.bold,
              ),
              textAlign: pw.TextAlign.center,
            ),
          ),
          pw.Expanded(
            flex: 2,
            child: pw.Text(
              element.description ?? '',
              style: pw.TextStyle(
                font: font,
                fontSize: 9,
                color: PdfColors.grey600,
              ),
            ),
          ),
        ],
      ),
    );
  }

  static PdfColor _getPdfColor(dynamic flutterColor) {
    // Convert Flutter color to PDF color
    final int value = flutterColor.value;
    final int r = (value >> 16) & 0xFF;
    final int g = (value >> 8) & 0xFF;
    final int b = value & 0xFF;
    return PdfColor.fromInt((0xFF << 24) | (r << 16) | (g << 8) | b);
  }
}

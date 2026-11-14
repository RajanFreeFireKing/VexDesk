import 'package:flutter/material.dart';

/// Exact Dark-Teal conversational AI color palette.
/// Exclusively uses pure dynamic container colors with zero background images.
class AppColors {
  // Main Canvas & Surfaces
  static const Color background = Color(0xFF07252B);      // Deep Dark Teal
  static const Color surface = Color(0xFF0A2D34);         // Deep Teal Surface
  static const Color surfaceElevated = Color(0xFF0E353D); // Muted Teal Slate (Input Card / Pills)
  static const Color surfaceHighlight = Color(0xFF16444D);// Lighter Teal (Pill Button / Hover)
  static const Color borderSubtle = Color(0xFF133E47);    // Dark Teal Border
  static const Color borderFocused = Color(0xFF22C55E);   // Emerald Focus

  // Accents
  static const Color primary = Color(0xFF22C55E);         // Emerald Green Accent / Glowing Dot
  static const Color primaryLight = Color(0xFF4ADE80);
  static const Color secondary = Color(0xFF38BDF8);       // Sky Cyan Accent
  static const Color actionButton = Color(0xFFF3EFE6);    // Warm Cream/Off-White Action Circle
  static const Color actionIcon = Color(0xFF07252B);      // Dark Teal Action Icon
  static const Color stopRed = Color(0xFFEF4444);         // Stop Generation Accent

  // Typography
  static const Color textPrimary = Color(0xFFF8FAFC);     // Crisp White
  static const Color textSecondary = Color(0xFF94A3B8);   // Muted Mint-Grey
  static const Color textMuted = Color(0xFF64748B);       // Darker Hint Grey

  // Chat Bubbles
  static const Color userBubble = Color(0xFF0E353D);
  static const Color assistantBubble = Color(0xFF0A2D34);
  static const Color codeBackground = Color(0xFF04191D);

  // Telemetry Badges
  static const Color badgeGreenBg = Color(0x2222C55E);
  static const Color badgeGreenText = Color(0xFF22C55E);
  static const Color badgeCyanBg = Color(0x2238BDF8);
  static const Color badgeCyanText = Color(0xFF38BDF8);
}

class AppTheme {
  static ThemeData get darkTealTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: AppColors.background,
      colorScheme: const ColorScheme.dark(
        primary: AppColors.primary,
        secondary: AppColors.secondary,
        surface: AppColors.surface,
        error: AppColors.stopRed,
        onPrimary: AppColors.actionIcon,
        onSurface: AppColors.textPrimary,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: AppColors.background,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        titleTextStyle: TextStyle(
          color: AppColors.textPrimary,
          fontSize: 18,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.2,
        ),
        iconTheme: IconThemeData(color: AppColors.textPrimary),
      ),
      cardTheme: CardTheme(
        color: AppColors.surfaceElevated,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(24),
          side: const BorderSide(color: AppColors.borderSubtle, width: 1),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.transparent,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        border: InputBorder.none,
        enabledBorder: InputBorder.none,
        focusedBorder: InputBorder.none,
        hintStyle: const TextStyle(color: AppColors.textMuted, fontSize: 15),
      ),
    );
  }
}

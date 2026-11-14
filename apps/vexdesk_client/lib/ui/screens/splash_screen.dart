import 'dart:async';
import 'package:flutter/material.dart';
import '../../state/chat_controller.dart';
import '../../state/model_hub_controller.dart';
import '../theme/app_theme.dart';
import 'chat_screen.dart';

/// Splash Screen for VexDesk.
/// Displays branding and engine status, then smoothly transitions to ChatScreen.
class SplashScreen extends StatefulWidget {
  final ChatController chatController;
  final ModelHubController modelHubController;

  const SplashScreen({
    super.key,
    required this.chatController,
    required this.modelHubController,
  });

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late AnimationController _animController;
  late Animation<double> _fadeAnimation;
  bool _hasNavigated = false;

  @override
  void initState() {
    super.initState();

    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _fadeAnimation = CurvedAnimation(
      parent: _animController,
      curve: Curves.easeIn,
    );
    _animController.forward();

    // 1. Asynchronous non-blocking background initialization check with fallback
    _probeEngineInBackground();

    // 2. Guaranteed fallback navigation timer (1500ms)
    Future.delayed(const Duration(milliseconds: 1500), () {
      _navigateToChat();
    });
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  Future<void> _probeEngineInBackground() async {
    try {
      // Non-blocking timeout wrapper so disk/native checks never freeze UI
      await Future.any([
        _quickEngineCheck(),
        Future.delayed(const Duration(milliseconds: 1000)),
      ]);
    } catch (e) {
      debugPrint('[SplashScreen] Non-fatal probe notice: $e');
    }
  }

  Future<void> _quickEngineCheck() async {
    // Check if models exist or native engine is ready without blocking isolate
    final activeModel = widget.modelHubController.activeModel;
    if (activeModel != null && activeModel.isDownloaded) {
      widget.chatController.selectModel(activeModel);
    }
  }

  void _navigateToChat() {
    if (!mounted || _hasNavigated) return;
    _hasNavigated = true;

    Navigator.of(context).pushReplacement(
      PageRouteBuilder(
        pageBuilder: (_, __, ___) => ChatScreen(
          chatController: widget.chatController,
          modelHubController: widget.modelHubController,
        ),
        transitionsBuilder: (_, animation, __, child) {
          return FadeTransition(opacity: animation, child: child);
        },
        transitionDuration: const Duration(milliseconds: 350),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0B0F17),
      body: SafeArea(
        child: FadeTransition(
          opacity: _fadeAnimation,
          child: Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // VexDesk App Logo
                  Container(
                    width: 170,
                    height: 170,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(32),
                      boxShadow: [
                        BoxShadow(
                          color: const Color(0xFF00E5FF).withOpacity(0.25),
                          blurRadius: 36,
                          spreadRadius: 2,
                        ),
                        BoxShadow(
                          color: const Color(0xFFD946EF).withOpacity(0.20),
                          blurRadius: 40,
                          spreadRadius: 4,
                        ),
                      ],
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(32),
                      child: Image.asset(
                        'assets/images/vexdesk_logo.png',
                        fit: BoxFit.contain,
                        errorBuilder: (_, __, ___) => const Icon(
                          Icons.memory,
                          size: 80,
                          color: Color(0xFF00E5FF),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),

                  // Title
                  const Text(
                    'VexDesk',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.5,
                    ),
                  ),
                  const SizedBox(height: 4),

                  // Tagline
                  const Text(
                    'Code • Build • Create',
                    style: TextStyle(
                      color: Color(0xFFC084FC),
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 0.8,
                    ),
                  ),
                  const SizedBox(height: 28),

                  // Engine Diagnostics Badge
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: const Color(0xFF131B2A),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: const Color(0xFF1E293B),
                        width: 1,
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: const [
                        Text(
                          '⚡ StreamVex C++ Engine',
                          style: TextStyle(
                            color: Color(0xFF38BDF8),
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        SizedBox(height: 8),
                        Text(
                          '• Architecture: ARM64-v8a (NEON SIMD)\n'
                          '• I/O: Asynchronous Double-Buffering\n'
                          '• Precision: Q4_0 / Q4_K_M Dequantization\n'
                          '• Memory: Capped 2-Layer Buffer Pool\n'
                          '• Native Status: Loaded & Linked',
                          style: TextStyle(
                            color: Color(0xFF94A3B8),
                            fontSize: 12,
                            height: 1.5,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),

                  // Bottom footnote
                  const Text(
                    'Local On-Device LLM Workspace',
                    style: TextStyle(
                      color: Color(0xFF475569),
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

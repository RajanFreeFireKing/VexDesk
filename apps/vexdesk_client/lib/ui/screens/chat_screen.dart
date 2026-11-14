import 'package:flutter/material.dart';
import '../../models/chat_message.dart';
import '../../models/llm_model.dart';
import '../../state/chat_controller.dart';
import '../../state/model_hub_controller.dart';
import '../theme/app_theme.dart';
import '../widgets/chat_bubble.dart';
import 'model_hub_screen.dart';

/// Redesigned Modern Dark-Teal Conversational AI Chat Screen.
/// Inspired by minimal assistant interfaces with instant launch,
/// top navigation bar with model pill, empty state typography,
/// and a docked floating multi-control card.
class ChatScreen extends StatefulWidget {
  final ChatController chatController;
  final ModelHubController modelHubController;

  const ChatScreen({
    super.key,
    required this.chatController,
    required this.modelHubController,
  });

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> with SingleTickerProviderStateMixin {
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final FocusNode _focusNode = FocusNode();

  late final AnimationController _headerAnimController;
  late final Animation<double> _headerFadeAnimation;
  late final Animation<Offset> _headerSlideAnimation;

  bool _hasText = false;
  bool _isListeningVoice = false;

  @override
  void initState() {
    super.initState();

    // 1. Smooth Fade-in + Slide animation for top header title on launch
    _headerAnimController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 650),
    );
    _headerFadeAnimation = CurvedAnimation(
      parent: _headerAnimController,
      curve: Curves.easeOut,
    );
    _headerSlideAnimation = Tween<Offset>(
      begin: const Offset(0.0, -0.3),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _headerAnimController,
      curve: Curves.easeOutCubic,
    ));
    _headerAnimController.forward();

    // 2. Controllers & Listeners
    _textController.addListener(_onTextChanged);
    widget.chatController.addListener(_onChatStateChanged);
    widget.modelHubController.addListener(_onModelHubChanged);
  }

  @override
  void dispose() {
    _headerAnimController.dispose();
    _textController.removeListener(_onTextChanged);
    _textController.dispose();
    _scrollController.dispose();
    _focusNode.dispose();
    widget.chatController.removeListener(_onChatStateChanged);
    widget.modelHubController.removeListener(_onModelHubChanged);
    super.dispose();
  }

  void _onTextChanged() {
    final hasContent = _textController.text.trim().isNotEmpty;
    if (hasContent != _hasText) {
      setState(() {
        _hasText = hasContent;
      });
    }
  }

  void _onChatStateChanged() {
    if (mounted) {
      setState(() {});
      if (widget.chatController.isGenerating) {
        _scrollToBottom();
      }
    }
  }

  void _onModelHubChanged() {
    if (mounted) {
      setState(() {});
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 150),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _handleSend() {
    final text = _textController.text.trim();
    if (text.isEmpty || widget.chatController.isGenerating) return;

    _textController.clear();
    setState(() {
      _hasText = false;
    });

    widget.chatController.sendMessage(text);
  }

  void _handleStop() {
    widget.chatController.stopGeneration();
  }

  void _handleNewChat() {
    widget.chatController.clearChat();
    _textController.clear();
    setState(() {
      _hasText = false;
    });
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Started fresh chat session'),
        duration: Duration(seconds: 1),
        backgroundColor: AppColors.surfaceElevated,
      ),
    );
  }

  void _openModelSelector() {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        side: BorderSide(color: AppColors.borderSubtle),
      ),
      builder: (ctx) => _buildModelSelectorSheet(ctx),
    );
  }

  void _openAttachmentSheet() {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        side: BorderSide(color: AppColors.borderSubtle),
      ),
      builder: (ctx) => _buildAttachmentSheet(ctx),
    );
  }

  void _openDiagnosticsSheet() {
    final stats = widget.chatController.stats;
    showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        side: BorderSide(color: AppColors.borderSubtle),
      ),
      builder: (ctx) => _buildDiagnosticsSheet(ctx, stats),
    );
  }

  void _toggleVoiceInput() {
    setState(() {
      _isListeningVoice = !_isListeningVoice;
    });

    if (_isListeningVoice) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Row(
            children: const [
              Icon(Icons.graphic_eq, color: AppColors.primary, size: 20),
              SizedBox(width: 10),
              Text('Voice listening active (Offline StreamVex STT)...'),
            ],
          ),
          duration: const Duration(seconds: 2),
          backgroundColor: AppColors.surfaceElevated,
          action: SnackBarAction(
            label: 'Dismiss',
            textColor: AppColors.primary,
            onPressed: () {
              setState(() {
                _isListeningVoice = false;
              });
            },
          ),
        ),
      );

      // Simulate voice transcription prompt insertion
      Future.delayed(const Duration(milliseconds: 1400), () {
        if (mounted && _isListeningVoice) {
          _textController.text = "Explain how StreamVex double-buffering streams GGUF models on mobile";
          setState(() {
            _hasText = true;
            _isListeningVoice = false;
          });
        }
      });
    }
  }

  void _openFullModelHub() {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ModelHubScreen(
          modelHubController: widget.modelHubController,
          chatController: widget.chatController,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final messages = widget.chatController.messages;
    final activeModel = widget.modelHubController.activeModel;
    final stats = widget.chatController.stats;

    return Scaffold(
      key: _scaffoldKey,
      backgroundColor: AppColors.background,
      drawer: _buildDrawer(activeModel, stats),
      body: SafeArea(
        child: Column(
          children: [
            // 1. Top Navigation Bar
            _buildTopNavBar(activeModel),

            // Optional Live Telemetry Banner (Toggled via speed button)
            if (widget.chatController.showDiagnostics)
              _buildTelemetryBanner(stats),

            // 2. Center Chat Body (Empty State vs Message Stream)
            Expanded(
              child: messages.isEmpty
                  ? _buildEmptyState()
                  : ListView.builder(
                      controller: _scrollController,
                      padding: const EdgeInsets.only(top: 8, bottom: 16),
                      itemCount: messages.length,
                      itemBuilder: (context, index) {
                        return ChatBubble(message: messages[index]);
                      },
                    ),
            ),

            // 3. Floating Bottom Multi-Control Dock
            _buildFloatingInputDock(stats),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // Top Navigation Bar
  // ===========================================================================
  Widget _buildTopNavBar(LlmModel? activeModel) {
    final modelDisplayName = activeModel?.name ?? 'SmolLM2-135M';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      color: AppColors.background,
      child: Row(
        children: [
          // Left: Hamburger Menu Button
          IconButton(
            key: const Key('btn_hamburger_menu'),
            icon: const Icon(Icons.menu, color: AppColors.textPrimary, size: 24),
            tooltip: 'Navigation Menu',
            onPressed: () {
              _scaffoldKey.currentState?.openDrawer();
            },
          ),
          const SizedBox(width: 4),

          // Center-Left: Animated Title "VexDesk"
          FadeTransition(
            opacity: _headerFadeAnimation,
            child: SlideTransition(
              position: _headerSlideAnimation,
              child: const Text(
                'VexDesk',
                style: TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 19,
                  fontWeight: FontWeight.bold,
                  letterSpacing: -0.2,
                ),
              ),
            ),
          ),

          const Spacer(),

          // Right-Center: Rounded Model Selector Pill
          GestureDetector(
            key: const Key('btn_model_selector_pill'),
            onTap: _openModelSelector,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: AppColors.surfaceElevated,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppColors.borderSubtle),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Glowing Emerald Dot
                  Container(
                    width: 8,
                    height: 8,
                    decoration: BoxDecoration(
                      color: AppColors.primary,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: AppColors.primary.withOpacity(0.6),
                          blurRadius: 6,
                          spreadRadius: 1,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 8),
                  ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 120),
                    child: Text(
                      modelDisplayName,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        color: AppColors.textPrimary,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  const SizedBox(width: 4),
                  const Icon(
                    Icons.arrow_drop_down,
                    color: AppColors.textSecondary,
                    size: 18,
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(width: 6),

          // Far Right: New Chat Square-Plus Button
          IconButton(
            key: const Key('btn_new_chat'),
            icon: const Icon(Icons.add_box_outlined, color: AppColors.textPrimary, size: 22),
            tooltip: 'New Chat',
            onPressed: _handleNewChat,
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // Optional Live Telemetry Strip
  // ===========================================================================
  Widget _buildTelemetryBanner(dynamic stats) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(
          bottom: BorderSide(color: AppColors.borderSubtle, width: 0.5),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              const Icon(Icons.bolt, color: AppColors.primary, size: 14),
              const SizedBox(width: 4),
              Text(
                'StreamVex: ${stats.isNativeDylib ? 'Native C++' : 'Double-Buffer'}',
                style: const TextStyle(color: AppColors.textSecondary, fontSize: 11),
              ),
            ],
          ),
          Row(
            children: [
              Text(
                'RAM: ${stats.ramUsageMb.toStringAsFixed(1)} MB',
                style: const TextStyle(
                  color: AppColors.primary,
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(width: 12),
              Text(
                'Speed: ${stats.tokensPerSec.toStringAsFixed(1)} t/s',
                style: const TextStyle(
                  color: AppColors.secondary,
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // Center Empty State
  // ===========================================================================
  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: const [
            Text(
              'How can I help you today?',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: AppColors.textPrimary,
                fontSize: 24,
                fontWeight: FontWeight.bold,
                letterSpacing: -0.3,
              ),
            ),
            SizedBox(height: 10),
            Text(
              'Ask a question, brainstorm ideas, or run local StreamVex models',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: AppColors.textSecondary,
                fontSize: 14,
                height: 1.4,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // Floating Bottom Multi-Control Dock (Exact Multi-Control Card)
  // ===========================================================================
  Widget _buildFloatingInputDock(dynamic stats) {
    final isGenerating = widget.chatController.isGenerating;

    return Container(
      margin: const EdgeInsets.only(left: 14, right: 14, bottom: 12, top: 4),
      decoration: BoxDecoration(
        color: AppColors.surfaceElevated,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: AppColors.borderSubtle, width: 1),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Top Row: Clean TextField
          Padding(
            padding: const EdgeInsets.only(left: 16, right: 16, top: 6),
            child: TextField(
              key: const Key('input_textfield'),
              controller: _textController,
              focusNode: _focusNode,
              maxLines: 4,
              minLines: 1,
              style: const TextStyle(
                color: AppColors.textPrimary,
                fontSize: 15,
                height: 1.4,
              ),
              decoration: const InputDecoration(
                hintText: 'Message VexDesk...',
                hintStyle: TextStyle(
                  color: AppColors.textMuted,
                  fontSize: 15,
                ),
                isDense: true,
                border: InputBorder.none,
                contentPadding: EdgeInsets.symmetric(vertical: 10),
              ),
              onSubmitted: (_) => _handleSend(),
            ),
          ),

          // Bottom Action Row inside Card
          Padding(
            padding: const EdgeInsets.only(left: 10, right: 10, bottom: 8, top: 2),
            child: Row(
              children: [
                // Left '+' Circle Button (File / Model Picker)
                GestureDetector(
                  key: const Key('btn_attach'),
                  onTap: _openAttachmentSheet,
                  child: Container(
                    width: 34,
                    height: 34,
                    decoration: BoxDecoration(
                      color: AppColors.surfaceHighlight,
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.add,
                      color: AppColors.textPrimary,
                      size: 20,
                    ),
                  ),
                ),
                const SizedBox(width: 8),

                // Compact Model/RAM Status Pill
                GestureDetector(
                  key: const Key('btn_status_pill'),
                  onTap: _openDiagnosticsSheet,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                    decoration: BoxDecoration(
                      color: AppColors.surfaceHighlight,
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(
                          Icons.auto_awesome,
                          color: AppColors.primary,
                          size: 13,
                        ),
                        const SizedBox(width: 6),
                        Text(
                          'StreamVex • ${stats.ramUsageMb.toStringAsFixed(0)}MB',
                          style: const TextStyle(
                            color: AppColors.textSecondary,
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(width: 4),
                        const Icon(
                          Icons.arrow_drop_down,
                          color: AppColors.textSecondary,
                          size: 16,
                        ),
                      ],
                    ),
                  ),
                ),

                const Spacer(),

                // 1. Diagnostics / Telemetry Toggle Button
                IconButton(
                  key: const Key('btn_diagnostics_toggle'),
                  icon: Icon(
                    Icons.speed_outlined,
                    color: widget.chatController.showDiagnostics
                        ? AppColors.primary
                        : AppColors.textSecondary,
                    size: 22,
                  ),
                  tooltip: 'Toggle Telemetry',
                  onPressed: () {
                    widget.chatController.toggleDiagnostics();
                  },
                ),

                // 2. Voice / Wake-Word Toggle Button
                IconButton(
                  key: const Key('btn_voice_input'),
                  icon: Icon(
                    Icons.graphic_eq,
                    color: _isListeningVoice ? AppColors.primary : AppColors.textSecondary,
                    size: 22,
                  ),
                  tooltip: 'Voice Input',
                  onPressed: _toggleVoiceInput,
                ),

                // 3. Primary Circular Action Button (Cream/Off-White Circle)
                GestureDetector(
                  key: const Key('btn_primary_action'),
                  onTap: () {
                    if (isGenerating) {
                      _handleStop();
                    } else if (_hasText) {
                      _handleSend();
                    } else {
                      _toggleVoiceInput();
                    }
                  },
                  child: Container(
                    width: 38,
                    height: 38,
                    decoration: BoxDecoration(
                      color: isGenerating ? AppColors.stopRed : AppColors.actionButton,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: (isGenerating ? AppColors.stopRed : AppColors.actionButton)
                              .withOpacity(0.2),
                          blurRadius: 8,
                          offset: const Offset(0, 2),
                        ),
                      ],
                    ),
                    child: Icon(
                      isGenerating
                          ? Icons.stop
                          : (_hasText ? Icons.arrow_upward : Icons.mic_none),
                      color: isGenerating ? Colors.white : AppColors.actionIcon,
                      size: 20,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // Navigation Drawer
  // ===========================================================================
  Widget _buildDrawer(LlmModel? activeModel, dynamic stats) {
    return Drawer(
      backgroundColor: AppColors.background,
      child: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Drawer Header
            Container(
              padding: const EdgeInsets.all(20),
              decoration: const BoxDecoration(
                color: AppColors.surface,
                border: Border(
                  bottom: BorderSide(color: AppColors.borderSubtle),
                ),
              ),
              child: Row(
                children: [
                  Container(
                    width: 44,
                    height: 44,
                    decoration: BoxDecoration(
                      color: AppColors.surfaceHighlight,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.memory, color: AppColors.primary, size: 24),
                  ),
                  const SizedBox(width: 14),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'VexDesk LLM',
                        style: TextStyle(
                          color: AppColors.textPrimary,
                          fontSize: 17,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        'StreamVex C++ Engine (arm64)',
                        style: TextStyle(
                          color: AppColors.textSecondary,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),

            // Sessions List
            Expanded(
              child: ListView(
                padding: const EdgeInsets.symmetric(vertical: 8),
                children: [
                  ListTile(
                    key: const Key('drawer_new_chat'),
                    leading: const Icon(Icons.add_comment_outlined, color: AppColors.primary),
                    title: const Text('New Chat', style: TextStyle(color: AppColors.textPrimary)),
                    onTap: () {
                      Navigator.pop(context);
                      _handleNewChat();
                    },
                  ),
                  ListTile(
                    key: const Key('drawer_current_session'),
                    leading: const Icon(Icons.chat_bubble_outline, color: AppColors.textSecondary),
                    title: const Text('Current Session', style: TextStyle(color: AppColors.textPrimary)),
                    subtitle: Text(
                      '${widget.chatController.messages.length} messages',
                      style: const TextStyle(color: AppColors.textMuted, fontSize: 11),
                    ),
                    onTap: () {
                      Navigator.pop(context);
                    },
                  ),
                  const Divider(color: AppColors.borderSubtle),
                  ListTile(
                    key: const Key('drawer_model_hub'),
                    leading: const Icon(Icons.hub_outlined, color: AppColors.secondary),
                    title: const Text('Model Hub', style: TextStyle(color: AppColors.textPrimary)),
                    subtitle: Text(
                      activeModel?.name ?? 'SmolLM2-135M',
                      style: const TextStyle(color: AppColors.textMuted, fontSize: 11),
                    ),
                    onTap: () {
                      Navigator.pop(context);
                      _openFullModelHub();
                    },
                  ),
                  ListTile(
                    key: const Key('drawer_telemetry'),
                    leading: const Icon(Icons.insights_rounded, color: AppColors.primary),
                    title: const Text('Engine Telemetry', style: TextStyle(color: AppColors.textPrimary)),
                    subtitle: Text(
                      '${stats.ramUsageMb.toStringAsFixed(1)} MB • Double-Buffering',
                      style: const TextStyle(color: AppColors.textMuted, fontSize: 11),
                    ),
                    onTap: () {
                      Navigator.pop(context);
                      _openDiagnosticsSheet();
                    },
                  ),
                  ListTile(
                    key: const Key('drawer_settings'),
                    leading: const Icon(Icons.settings_outlined, color: AppColors.textSecondary),
                    title: const Text('Settings & Storage', style: TextStyle(color: AppColors.textPrimary)),
                    onTap: () {
                      Navigator.pop(context);
                      _openDiagnosticsSheet();
                    },
                  ),
                ],
              ),
            ),

            // Drawer Footer
            Container(
              padding: const EdgeInsets.all(16),
              color: AppColors.surface,
              child: Row(
                children: [
                  Container(
                    width: 8,
                    height: 8,
                    decoration: const BoxDecoration(
                      color: AppColors.primary,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 8),
                  const Text(
                    'Engine Status: Online (Local)',
                    style: TextStyle(color: AppColors.textSecondary, fontSize: 12),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // Model Selector Modal Bottom Sheet
  // ===========================================================================
  Widget _buildModelSelectorSheet(BuildContext ctx) {
    final catalog = widget.modelHubController.catalog;
    final activeId = widget.modelHubController.activeModel?.id;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Select Active Model',
                style: TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              TextButton(
                onPressed: () {
                  Navigator.pop(ctx);
                  _openFullModelHub();
                },
                child: const Text('Browse Hub', style: TextStyle(color: AppColors.primary)),
              ),
            ],
          ),
          const SizedBox(height: 12),
          ConstrainedBox(
            constraints: const BoxConstraints(maxHeight: 280),
            child: ListView.separated(
              shrinkWrap: true,
              itemCount: catalog.length,
              separatorBuilder: (_, __) => const SizedBox(height: 8),
              itemBuilder: (context, idx) {
                final model = catalog[idx];
                final isSelected = model.id == activeId;

                return InkWell(
                  onTap: () async {
                    Navigator.pop(ctx);
                    await widget.modelHubController.loadModelIntoEngine(model.id);
                    widget.chatController.updateActiveModel(model);
                  },
                  borderRadius: BorderRadius.circular(14),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                    decoration: BoxDecoration(
                      color: isSelected ? AppColors.surfaceHighlight : AppColors.surfaceElevated,
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(
                        color: isSelected ? AppColors.primary : AppColors.borderSubtle,
                      ),
                    ),
                    child: Row(
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            color: isSelected ? AppColors.primary : AppColors.textMuted,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                model.name,
                                style: const TextStyle(
                                  color: AppColors.textPrimary,
                                  fontSize: 14,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                              Text(
                                '${model.parameterSize} • ${model.quantization} • ${model.fileSize}',
                                style: const TextStyle(color: AppColors.textSecondary, fontSize: 11),
                              ),
                            ],
                          ),
                        ),
                        if (isSelected)
                          const Icon(Icons.check, color: AppColors.primary, size: 18),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // Attachment & File Picker Bottom Sheet
  // ===========================================================================
  Widget _buildAttachmentSheet(BuildContext ctx) {
    return Container(
      padding: const EdgeInsets.all(20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Add Context or Model',
            style: TextStyle(
              color: AppColors.textPrimary,
              fontSize: 17,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          ListTile(
            leading: const Icon(Icons.folder_open, color: AppColors.primary),
            title: const Text('Import Local GGUF Model', style: TextStyle(color: AppColors.textPrimary)),
            subtitle: const Text('Select .gguf weights file from storage', style: TextStyle(color: AppColors.textSecondary, fontSize: 12)),
            onTap: () {
              Navigator.pop(ctx);
              widget.modelHubController.importLocalModel(
                name: 'Custom-Model-Q4_K_M',
                filePath: '/sdcard/Download/model.gguf',
                parameterSize: '0.5B',
                quantization: 'Q4_K_M',
                fileSize: '350 MB',
                estimatedRamMb: 48.0,
              );
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Imported custom model from /sdcard'),
                  backgroundColor: AppColors.surfaceElevated,
                ),
              );
            },
          ),
          ListTile(
            leading: const Icon(Icons.description_outlined, color: AppColors.secondary),
            title: const Text('Attach Text / Code Document', style: TextStyle(color: AppColors.textPrimary)),
            subtitle: const Text('Inject file content into LLM context window', style: TextStyle(color: AppColors.textSecondary, fontSize: 12)),
            onTap: () {
              Navigator.pop(ctx);
              _textController.text = "[Document Context Attached: main.cpp]\n" + _textController.text;
              setState(() {
                _hasText = true;
              });
            },
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // Quick Diagnostics & Quantization Sheet
  // ===========================================================================
  Widget _buildDiagnosticsSheet(BuildContext ctx, dynamic stats) {
    return Container(
      padding: const EdgeInsets.all(22),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: const [
              Icon(Icons.tune, color: AppColors.primary, size: 20),
              SizedBox(width: 8),
              Text(
                'StreamVex Engine Telemetry',
                style: TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 17,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _diagItem('Architecture', 'ARM64-v8a (SIMD NEON Vectorized)'),
          _diagItem('I/O Layer Streaming', 'Asynchronous Double-Buffering (Capped)'),
          _diagItem('Resident RAM', '${stats.ramUsageMb.toStringAsFixed(1)} MB'),
          _diagItem('Generation Throughput', '${stats.tokensPerSec.toStringAsFixed(1)} tokens/sec'),
          _diagItem('Total Generated Tokens', '${stats.totalTokens} tokens'),
          _diagItem('Native Dynamic Library', stats.isNativeDylib ? 'Connected (libstreamvex.so)' : 'Simulator Fallback'),
          const SizedBox(height: 14),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.surfaceHighlight,
                foregroundColor: AppColors.primary,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              ),
              onPressed: () {
                Navigator.pop(ctx);
                _openFullModelHub();
              },
              child: const Text('Manage Models in Hub'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _diagItem(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: AppColors.textSecondary, fontSize: 12)),
          Flexible(
            child: Text(
              value,
              textAlign: TextAlign.end,
              style: const TextStyle(
                color: AppColors.textPrimary,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

/// Clean bottom input bar for submitting prompts and stopping generation.
class ChatInputBar extends StatefulWidget {
  final bool isGenerating;
  final ValueChanged<String> onSend;
  final VoidCallback onStop;

  const ChatInputBar({
    super.key,
    required this.isGenerating,
    required this.onSend,
    required this.onStop,
  });

  @override
  State<ChatInputBar> createState() => _ChatInputBarState();
}

class _ChatInputBarState extends State<ChatInputBar> {
  final TextEditingController _controller = TextEditingController();

  void _handleSubmitted() {
    final text = _controller.text.trim();
    if (text.isNotEmpty) {
      widget.onSend(text);
      _controller.clear();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 16),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(
          top: BorderSide(color: AppColors.borderSubtle, width: 1),
        ),
      ),
      child: SafeArea(
        top: false,
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            // Text Input Field
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  color: AppColors.surfaceElevated,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: AppColors.borderSubtle),
                ),
                child: TextField(
                  controller: _controller,
                  minLines: 1,
                  maxLines: 5,
                  textInputAction: TextInputAction.send,
                  onSubmitted: (_) => _handleSubmitted(),
                  style: const TextStyle(
                    color: AppColors.textPrimary,
                    fontSize: 14.5,
                  ),
                  decoration: const InputDecoration(
                    hintText: 'Ask StreamVex or send test prompt...',
                    hintStyle: TextStyle(color: AppColors.textMuted, fontSize: 14),
                    contentPadding: EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 12,
                    ),
                    border: InputBorder.none,
                    enabledBorder: InputBorder.none,
                    focusedBorder: InputBorder.none,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 10),

            // Submit or Stop Button
            AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: widget.isGenerating ? AppColors.rose : AppColors.primary,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: (widget.isGenerating ? AppColors.rose : AppColors.primary)
                        .withOpacity(0.35),
                    blurRadius: 10,
                    offset: const Offset(0, 3),
                  ),
                ],
              ),
              child: IconButton(
                padding: EdgeInsets.zero,
                icon: Icon(
                  widget.isGenerating ? Icons.stop_rounded : Icons.arrow_upward_rounded,
                  color: Colors.white,
                  size: 22,
                ),
                onPressed: () {
                  if (widget.isGenerating) {
                    widget.onStop();
                  } else {
                    _handleSubmitted();
                  }
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

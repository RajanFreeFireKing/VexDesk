import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../../models/chat_message.dart';
import '../theme/app_theme.dart';

/// Clean dynamic chat bubble widget for user prompts and streaming assistant outputs.
/// Strictly uses dynamic container styling with zero background images.
class ChatBubble extends StatelessWidget {
  final ChatMessage message;

  const ChatBubble({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    if (message.isUser) {
      return _buildUserBubble(context);
    } else {
      return _buildAssistantBubble(context);
    }
  }

  Widget _buildUserBubble(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: AppColors.userBubble,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(18),
                  topRight: Radius.circular(18),
                  bottomLeft: Radius.circular(18),
                  bottomRight: Radius.circular(4),
                ),
                boxShadow: [
                  BoxShadow(
                    color: AppColors.userBubble.withOpacity(0.25),
                    blurRadius: 8,
                    offset: const Offset(0, 3),
                  ),
                ],
              ),
              child: Text(
                message.text,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  height: 1.45,
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),
          CircleAvatar(
            radius: 14,
            backgroundColor: AppColors.surfaceElevated,
            child: const Icon(Icons.person, size: 16, color: AppColors.textSecondary),
          ),
        ],
      ),
    );
  }

  Widget _buildAssistantBubble(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          CircleAvatar(
            radius: 14,
            backgroundColor: AppColors.primary.withOpacity(0.2),
            child: const Icon(
              Icons.auto_awesome,
              size: 15,
              color: AppColors.primaryLight,
            ),
          ),
          const SizedBox(width: 10),
          Flexible(
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppColors.assistantBubble,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(4),
                  topRight: Radius.circular(18),
                  bottomLeft: Radius.circular(18),
                  bottomRight: Radius.circular(18),
                ),
                border: Border.all(color: AppColors.borderSubtle, width: 1),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Message Body
                  if (message.text.isEmpty && message.isStreaming)
                    Row(
                      children: [
                        SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: AppColors.primaryLight,
                          ),
                        ),
                        const SizedBox(width: 10),
                        const Text(
                          'Evaluating prompt...',
                          style: TextStyle(
                            color: AppColors.textMuted,
                            fontSize: 13,
                            fontStyle: FontStyle.italic,
                          ),
                        ),
                      ],
                    )
                  else
                    SelectableText(
                      message.text,
                      style: const TextStyle(
                        color: AppColors.textPrimary,
                        fontSize: 14.5,
                        height: 1.5,
                        letterSpacing: -0.1,
                      ),
                    ),

                  // Streaming cursor
                  if (message.isStreaming && message.text.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(top: 4),
                      child: Container(
                        width: 7,
                        height: 14,
                        color: AppColors.secondary,
                      ),
                    ),

                  // Message Diagnostics Footer
                  if (!message.isStreaming && message.text.isNotEmpty) ...[
                    const SizedBox(height: 12),
                    Divider(color: AppColors.borderSubtle.withOpacity(0.5), height: 1),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        // Telemetry summary badge
                        Wrap(
                          spacing: 8,
                          children: [
                            if (message.tokensPerSec != null)
                              _MetricTag(
                                icon: Icons.speed,
                                text: '${message.tokensPerSec!.toStringAsFixed(1)} t/s',
                                color: AppColors.secondary,
                              ),
                            if (message.totalTokens != null)
                              _MetricTag(
                                icon: Icons.token,
                                text: '${message.totalTokens} tokens',
                                color: AppColors.primaryLight,
                              ),
                            if (message.generationTimeMs != null)
                              _MetricTag(
                                icon: Icons.timer_outlined,
                                text: '${(message.generationTimeMs! / 1000).toStringAsFixed(1)}s',
                                color: AppColors.emerald,
                              ),
                          ],
                        ),

                        // Copy Button
                        IconButton(
                          icon: const Icon(Icons.copy, size: 14),
                          color: AppColors.textMuted,
                          padding: EdgeInsets.zero,
                          constraints: const BoxConstraints(),
                          splashRadius: 16,
                          tooltip: 'Copy response',
                          onPressed: () {
                            Clipboard.setData(ClipboardData(text: message.text));
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text('Copied response to clipboard'),
                                duration: Duration(seconds: 1),
                                behavior: SnackBarBehavior.floating,
                              ),
                            );
                          },
                        ),
                      ],
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MetricTag extends StatelessWidget {
  final IconData icon;
  final String text;
  final Color color;

  const _MetricTag({
    required this.icon,
    required this.text,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 12, color: color),
        const SizedBox(width: 4),
        Text(
          text,
          style: TextStyle(
            color: color,
            fontSize: 11,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }
}

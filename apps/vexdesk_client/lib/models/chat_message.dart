/// Represents an individual chat message in the conversation.
class ChatMessage {
  final String id;
  final String text;
  final bool isUser;
  final DateTime timestamp;
  final bool isStreaming;
  final double? tokensPerSec;
  final int? totalTokens;
  final int? generationTimeMs;

  const ChatMessage({
    required this.id,
    required this.text,
    required this.isUser,
    required this.timestamp,
    this.isStreaming = false,
    this.tokensPerSec,
    this.totalTokens,
    this.generationTimeMs,
  });

  ChatMessage copyWith({
    String? id,
    String? text,
    bool? isUser,
    DateTime? timestamp,
    bool? isStreaming,
    double? tokensPerSec,
    int? totalTokens,
    int? generationTimeMs,
  }) {
    return ChatMessage(
      id: id ?? this.id,
      text: text ?? this.text,
      isUser: isUser ?? this.isUser,
      timestamp: timestamp ?? this.timestamp,
      isStreaming: isStreaming ?? this.isStreaming,
      tokensPerSec: tokensPerSec ?? this.tokensPerSec,
      totalTokens: totalTokens ?? this.totalTokens,
      generationTimeMs: generationTimeMs ?? this.generationTimeMs,
    );
  }
}

import 'package:flutter/material.dart';
import 'state/chat_controller.dart';
import 'state/model_hub_controller.dart';
import 'ui/screens/chat_screen.dart';
import 'ui/theme/app_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  final modelHubController = ModelHubController();
  final chatController = ChatController();

  runApp(VexDeskApp(
    chatController: chatController,
    modelHubController: modelHubController,
  ));
}

class VexDeskApp extends StatelessWidget {
  final ChatController chatController;
  final ModelHubController modelHubController;

  const VexDeskApp({
    super.key,
    required this.chatController,
    required this.modelHubController,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'VexDesk',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkTealTheme,
      // Instant Launch: opens straight into ChatScreen with zero splash delay
      home: ChatScreen(
        chatController: chatController,
        modelHubController: modelHubController,
      ),
      routes: {
        '/chat': (_) => ChatScreen(
              chatController: chatController,
              modelHubController: modelHubController,
            ),
      },
    );
  }
}

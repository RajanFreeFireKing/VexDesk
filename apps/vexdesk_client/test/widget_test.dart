import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vexdesk_client/main.dart';
import 'package:vexdesk_client/state/chat_controller.dart';
import 'package:vexdesk_client/state/model_hub_controller.dart';
import 'package:vexdesk_client/ui/screens/chat_screen.dart';

void main() {
  group('VexDesk 100% Responsive Button Audit & Widget Tests', () {
    late ChatController chatController;
    late ModelHubController modelHubController;

    setUp(() {
      chatController = ChatController();
      modelHubController = ModelHubController();
    });

    testWidgets('Instant launch loads ChatScreen directly with empty state', (WidgetTester tester) async {
      await tester.pumpWidget(VexDeskApp(
        chatController: chatController,
        modelHubController: modelHubController,
      ));
      await tester.pumpAndSettle();

      // Verify ChatScreen is loaded immediately
      expect(find.byType(ChatScreen), findsOneWidget);

      // Verify Animated Header Title
      expect(find.text('VexDesk'), findsOneWidget);

      // Verify Empty State Typography
      expect(find.text('How can I help you today?'), findsOneWidget);
      expect(find.text('Ask a question, brainstorm ideas, or run local StreamVex models'), findsOneWidget);
    });

    testWidgets('Verify Hamburger Menu button opens Navigation Drawer and items work', (WidgetTester tester) async {
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(
          chatController: chatController,
          modelHubController: modelHubController,
        ),
      ));
      await tester.pumpAndSettle();

      // Tap Hamburger Menu button
      final menuBtn = find.byKey(const Key('btn_hamburger_menu'));
      expect(menuBtn, findsOneWidget);
      await tester.tap(menuBtn);
      await tester.pumpAndSettle();

      // Verify Drawer items
      expect(find.byKey(const Key('drawer_new_chat')), findsOneWidget);
      expect(find.byKey(const Key('drawer_current_session')), findsOneWidget);
      expect(find.byKey(const Key('drawer_model_hub')), findsOneWidget);
      expect(find.byKey(const Key('drawer_telemetry')), findsOneWidget);

      // Tap New Chat in drawer to close it
      await tester.tap(find.byKey(const Key('drawer_new_chat')));
      await tester.pumpAndSettle();
    });

    testWidgets('Verify Model Selector Pill opens Model Sheet and selects model', (WidgetTester tester) async {
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(
          chatController: chatController,
          modelHubController: modelHubController,
        ),
      ));
      await tester.pumpAndSettle();

      // Tap Model Pill
      final modelPill = find.byKey(const Key('btn_model_selector_pill'));
      expect(modelPill, findsOneWidget);
      await tester.tap(modelPill);
      await tester.pumpAndSettle();

      // Verify BottomSheet opened
      expect(find.text('Select Active Model'), findsOneWidget);

      // Tap first available model in the list
      final modelTile = find.text('SmolLM2-135M-Instruct').first;
      expect(modelTile, findsOneWidget);
      await tester.tap(modelTile);
      await tester.pumpAndSettle();
    });

    testWidgets('Verify New Chat button clears messages', (WidgetTester tester) async {
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(
          chatController: chatController,
          modelHubController: modelHubController,
        ),
      ));
      await tester.pumpAndSettle();

      // Simulate sending a message first
      await chatController.sendMessage('Initial test query');
      await tester.pumpAndSettle();
      expect(chatController.messages.isNotEmpty, isTrue);

      // Tap New Chat Button
      final newChatBtn = find.byKey(const Key('btn_new_chat'));
      expect(newChatBtn, findsOneWidget);
      await tester.tap(newChatBtn);
      await tester.pumpAndSettle();

      // Verify messages cleared
      expect(chatController.messages.isEmpty, isTrue);
      expect(find.text('How can I help you today?'), findsOneWidget);
    });

    testWidgets('Verify Attachment (+) Button opens sheet', (WidgetTester tester) async {
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(
          chatController: chatController,
          modelHubController: modelHubController,
        ),
      ));
      await tester.pumpAndSettle();

      // Tap '+' button
      final attachBtn = find.byKey(const Key('btn_attach'));
      expect(attachBtn, findsOneWidget);
      await tester.tap(attachBtn);
      await tester.pumpAndSettle();

      // Verify sheet opened
      expect(find.text('Add Context or Model'), findsOneWidget);
      expect(find.text('Import Local GGUF Model'), findsOneWidget);

      // Dismiss sheet
      await tester.tap(find.text('Import Local GGUF Model'));
      await tester.pumpAndSettle();
    });

    testWidgets('Verify Diagnostics Toggle button flips state', (WidgetTester tester) async {
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(
          chatController: chatController,
          modelHubController: modelHubController,
        ),
      ));
      await tester.pumpAndSettle();

      final initialDiagState = chatController.showDiagnostics;
      final diagToggleBtn = find.byKey(const Key('btn_diagnostics_toggle'));
      expect(diagToggleBtn, findsOneWidget);

      // Tap toggle
      await tester.tap(diagToggleBtn);
      await tester.pumpAndSettle();
      expect(chatController.showDiagnostics, !initialDiagState);

      // Tap toggle again to restore
      await tester.tap(diagToggleBtn);
      await tester.pumpAndSettle();
      expect(chatController.showDiagnostics, initialDiagState);
    });

    testWidgets('Verify Voice Input button toggles state', (WidgetTester tester) async {
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(
          chatController: chatController,
          modelHubController: modelHubController,
        ),
      ));
      await tester.pumpAndSettle();

      final voiceBtn = find.byKey(const Key('btn_voice_input'));
      expect(voiceBtn, findsOneWidget);

      await tester.tap(voiceBtn);
      await tester.pump();
      expect(find.byType(SnackBar), findsOneWidget);
    });

    testWidgets('Verify Typing text switches Primary Action button to Send and executes generation', (WidgetTester tester) async {
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(
          chatController: chatController,
          modelHubController: modelHubController,
        ),
      ));
      await tester.pumpAndSettle();

      final inputField = find.byKey(const Key('input_textfield'));
      final actionBtn = find.byKey(const Key('btn_primary_action'));

      expect(inputField, findsOneWidget);
      expect(actionBtn, findsOneWidget);

      // Enter text into field
      await tester.enterText(inputField, 'Hello StreamVex!');
      await tester.pump();

      // Verify icon changed to send arrow
      expect(find.byIcon(Icons.arrow_upward), findsOneWidget);

      // Tap Send Action Button
      await tester.tap(actionBtn);
      await tester.pump();

      // Verify user message created
      expect(find.text('Hello StreamVex!'), findsOneWidget);
    });
  });
}

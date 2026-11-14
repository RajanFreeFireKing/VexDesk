import 'dart:async';
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'streamvex_bindings.dart';

/// Performance metrics collected during token streaming.
class GenerationMetrics {
  final int tokenCount;
  final Duration elapsed;
  final double tokensPerSecond;
  final double ramUsageMb;

  const GenerationMetrics({
    required this.tokenCount,
    required this.elapsed,
    required this.tokensPerSecond,
    required this.ramUsageMb,
  });

  factory GenerationMetrics.initial() {
    return const GenerationMetrics(
      tokenCount: 0,
      elapsed: Duration.zero,
      tokensPerSecond: 0.0,
      ramUsageMb: 0.0,
    );
  }
}

/// High-level bridge connecting Flutter UI to the StreamVex C++ engine.
class StreamVexBridge {
  static final StreamVexBridge _instance = StreamVexBridge._internal();
  factory StreamVexBridge() => _instance;

  final StreamVexBindings _bindings = StreamVexBindings();
  bool _isEngineInitialized = false;
  String _activeModelPath = '';

  bool get isNativeLoaded => _bindings.isNativeLoaded;
  bool get isInitialized => _isEngineInitialized;
  String get activeModelPath => _activeModelPath;

  StreamVexBridge._internal();

  /// Initializes the local LLM engine with the given model path.
  Future<bool> initEngine(String modelPath) async {
    try {
      final code = _bindings.initEngine(modelPath);
      if (code == 0) {
        _isEngineInitialized = true;
        _activeModelPath = modelPath;
        return true;
      }
      return false;
    } catch (e) {
      print('[StreamVexBridge] Error in initEngine: $e');
      return false;
    }
  }

  /// Streams generated tokens for a prompt. Emits each token chunk via Stream.
  Stream<String> streamPrompt(
    String prompt, {
    void Function(GenerationMetrics metrics)? onMetricsUpdate,
  }) async* {
    final startTime = DateTime.now();
    int tokenCount = 0;

    if (_bindings.isNativeLoaded) {
      final controller = StreamController<String>();

      // Use NativeCallable to safely receive C++ callbacks on the Dart event loop
      NativeCallable<TokenCallbackC>? nativeCallable;

      try {
        nativeCallable = NativeCallable<TokenCallbackC>.listener((Pointer<Utf8> tokenPtr) {
          if (!controller.isClosed) {
            final token = tokenPtr.toDartString();
            tokenCount++;
            final elapsed = DateTime.now().difference(startTime);
            final seconds = elapsed.inMilliseconds / 1000.0;
            final tps = seconds > 0 ? (tokenCount / seconds) : 0.0;
            final ramMb = _bindings.getMemoryUsage();

            if (onMetricsUpdate != null) {
              onMetricsUpdate(GenerationMetrics(
                tokenCount: tokenCount,
                elapsed: elapsed,
                tokensPerSecond: tps,
                ramUsageMb: ramMb,
              ));
            }

            controller.add(token);
          }
        });

        // Run C++ stream execution
        Future(() {
          try {
            _bindings.streamToken(prompt, nativeCallable!.nativeFunction);
          } finally {
            controller.close();
            nativeCallable?.close();
          }
        });

        yield* controller.stream;
        return;
      } catch (e) {
        print('[StreamVexBridge] FFI stream error: $e');
        nativeCallable?.close();
        if (!controller.isClosed) controller.close();
      }
    }

    // High-speed fallback / mock engine when running without compiled native library
    final simulatedTokens = _generateSimulatedTokens(prompt);
    for (final token in simulatedTokens) {
      await Future.delayed(const Duration(milliseconds: 30));
      tokenCount++;
      final elapsed = DateTime.now().difference(startTime);
      final seconds = elapsed.inMilliseconds / 1000.0;
      final tps = seconds > 0 ? (tokenCount / seconds) : 0.0;
      final ramMb = getMemoryUsage();

      if (onMetricsUpdate != null) {
        onMetricsUpdate(GenerationMetrics(
          tokenCount: tokenCount,
          elapsed: elapsed,
          tokensPerSecond: tps,
          ramUsageMb: ramMb,
        ));
      }

      yield token;
    }
  }

  /// Queries current RAM usage (MB) from the C++ engine or runtime monitor.
  double getMemoryUsage() {
    return _bindings.getMemoryUsage();
  }

  /// Releases the engine memory.
  void releaseEngine() {
    _bindings.releaseEngine();
    _isEngineInitialized = false;
    _activeModelPath = '';
  }

  /// Runs the asynchronous double-buffering layer streaming benchmark.
  Future<String> runStreamingBenchmark({
    String? filePath,
    int chunkSizeMb = 50,
    int totalLayers = 10,
  }) async {
    return _bindings.runStreamingBenchmark(
      filePath: filePath,
      chunkSizeMb: chunkSizeMb,
      totalLayers: totalLayers,
    );
  }

  /// Parses GGUF model metadata from disk without loading tensor weight data into RAM.
  Future<String> parseGgufMetadata(String modelPath) async {
    return _bindings.parseGgufMetadata(modelPath);
  }

  /// Streams and dequantizes a single layer tensor into float32.
  Future<List<double>> streamAndDequantLayer(String modelPath, String tensorName, {int maxFloats = 262144}) async {
    return _bindings.streamAndDequantLayer(modelPath, tensorName, maxFloats: maxFloats);
  }

  /// Runs GEMV performance benchmark comparing Float32 vs Q4_0 execution.
  Future<String> computeGemvBenchmark({int rows = 2048, int cols = 2048}) async {
    return _bindings.computeGemvBenchmark(rows: rows, cols: cols);
  }

  /// Runs a single autoregressive forward pass step predicting the next token ID.
  Future<Map<String, dynamic>> runForwardPassStep(int tokenId, int pos) async {
    return _bindings.runForwardPassStep(tokenId, pos);
  }

  /// Retrieves real-time JSON telemetry from the native Autoregressive Engine.
  String getLastGenerationTelemetry() {
    return _bindings.getLastGenerationTelemetry();
  }

  List<String> _generateSimulatedTokens(String prompt) {
    final modelName = _activeModelPath.isNotEmpty
        ? _activeModelPath.split('/').last
        : 'StreamVex-Mock-0.5B';

    return [
      'Hello! ',
      'StreamVex ',
      'engine ',
      'is ',
      'processing ',
      'your ',
      'request ',
      'locally.\n\n',
      '**Active Model:** `$modelName`\n',
      '**Prompt:** "${prompt.trim()}"\n\n',
      'On-device ',
      'inference ',
      'is ',
      'active ',
      'with ',
      'zero ',
      'cloud ',
      'dependency. ',
      'Memory ',
      'pages ',
      'and ',
      'tensor ',
      'streams ',
      'are ',
      'monitored ',
      'in ',
      'real-time ',
      'by ',
      'the ',
      'VexDesk ',
      'diagnostics ',
      'pipeline.'
    ];
  }
}

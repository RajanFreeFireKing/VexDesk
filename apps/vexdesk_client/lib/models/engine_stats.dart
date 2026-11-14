/// Live telemetry data for the top diagnostics bar and engine monitor.
class EngineStats {
  final double tokensPerSec;
  final double ramUsageMb;
  final String modelName;
  final bool isGenerating;
  final int totalTokens;
  final int latencyMs;
  final bool isNativeDylib;

  const EngineStats({
    this.tokensPerSec = 0.0,
    this.ramUsageMb = 32.2,
    this.modelName = 'SmolLM2-135M-Instruct',
    this.isGenerating = false,
    this.totalTokens = 0,
    this.latencyMs = 0,
    this.isNativeDylib = false,
  });

  EngineStats copyWith({
    double? tokensPerSec,
    double? ramUsageMb,
    String? modelName,
    bool? isGenerating,
    int? totalTokens,
    int? latencyMs,
    bool? isNativeDylib,
  }) {
    return EngineStats(
      tokensPerSec: tokensPerSec ?? this.tokensPerSec,
      ramUsageMb: ramUsageMb ?? this.ramUsageMb,
      modelName: modelName ?? this.modelName,
      isGenerating: isGenerating ?? this.isGenerating,
      totalTokens: totalTokens ?? this.totalTokens,
      latencyMs: latencyMs ?? this.latencyMs,
      isNativeDylib: isNativeDylib ?? this.isNativeDylib,
    );
  }
}

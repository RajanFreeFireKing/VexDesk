enum ModelSizeCategory {
  small('0.5B', 'Small (0.5B)', 'Ultra-fast, lowest memory footprint (sub-500MB)'),
  medium('3B', 'Medium (3B)', 'Balanced quality & speed, ideal for mobile/edge'),
  large('7B+', 'Large (7B+)', 'High reasoning, requires 4GB+ available RAM'),
  custom('Custom', 'Local Storage Import', 'User-imported .gguf weights');

  final String code;
  final String label;
  final String description;
  const ModelSizeCategory(this.code, this.label, this.description);
}

/// Represents an LLM model available in the VexDesk Model Hub.
class LlmModel {
  final String id;
  final String name;
  final ModelSizeCategory category;
  final String parameterSize;
  final String quantization;
  final String fileSize;
  final double estimatedRamMb;
  final String downloadUrl;
  final String? localFilePath;
  final bool isDownloaded;
  final bool isDownloading;
  final double downloadProgress; // 0.0 to 1.0
  final bool isLoaded;

  const LlmModel({
    required this.id,
    required this.name,
    required this.category,
    required this.parameterSize,
    required this.quantization,
    required this.fileSize,
    required this.estimatedRamMb,
    required this.downloadUrl,
    this.localFilePath,
    this.isDownloaded = false,
    this.isDownloading = false,
    this.downloadProgress = 0.0,
    this.isLoaded = false,
  });

  LlmModel copyWith({
    String? id,
    String? name,
    ModelSizeCategory? category,
    String? parameterSize,
    String? quantization,
    String? fileSize,
    double? estimatedRamMb,
    String? downloadUrl,
    String? localFilePath,
    bool? isDownloaded,
    bool? isDownloading,
    double? downloadProgress,
    bool? isLoaded,
  }) {
    return LlmModel(
      id: id ?? this.id,
      name: name ?? this.name,
      category: category ?? this.category,
      parameterSize: parameterSize ?? this.parameterSize,
      quantization: quantization ?? this.quantization,
      fileSize: fileSize ?? this.fileSize,
      estimatedRamMb: estimatedRamMb ?? this.estimatedRamMb,
      downloadUrl: downloadUrl ?? this.downloadUrl,
      localFilePath: localFilePath ?? this.localFilePath,
      isDownloaded: isDownloaded ?? this.isDownloaded,
      isDownloading: isDownloading ?? this.isDownloading,
      downloadProgress: downloadProgress ?? this.downloadProgress,
      isLoaded: isLoaded ?? this.isLoaded,
    );
  }

  /// Preset models categorized for Small (0.5B), Medium (3B), and Large (7B+)
  static List<LlmModel> get defaultCatalog => [
    const LlmModel(
      id: 'smollm2-135m',
      name: 'SmolLM2-135M-Instruct',
      category: ModelSizeCategory.small,
      parameterSize: '0.13B',
      quantization: 'Q4_K_M',
      fileSize: '101 MB',
      estimatedRamMb: 32.0,
      downloadUrl: 'https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF',
      isDownloaded: true,
      isLoaded: true,
      localFilePath: 'models/SmolLM2-135M-Instruct-Q4_K_M.gguf',
    ),
    const LlmModel(
      id: 'qwen-0.5b',
      name: 'Qwen2.5-0.5B-Instruct',
      category: ModelSizeCategory.small,
      parameterSize: '0.49B',
      quantization: 'Q4_K_M',
      fileSize: '392 MB',
      estimatedRamMb: 420.0,
      downloadUrl: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF',
      isDownloaded: false,
      isLoaded: false,
      localFilePath: 'models/qwen2.5-0.5b-instruct-q4_k_m.gguf',
    ),
    const LlmModel(
      id: 'smollm-360m',
      name: 'SmolLM2-360M-Instruct',
      category: ModelSizeCategory.small,
      parameterSize: '0.36B',
      quantization: 'Q8_0',
      fileSize: '380 MB',
      estimatedRamMb: 350.0,
      downloadUrl: 'https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF',
      isDownloaded: false,
    ),
    const LlmModel(
      id: 'llama-3.2-3b',
      name: 'Llama-3.2-3B-Instruct',
      category: ModelSizeCategory.medium,
      parameterSize: '3.21B',
      quantization: 'Q4_K_M',
      fileSize: '1.92 GB',
      estimatedRamMb: 2150.0,
      downloadUrl: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF',
      isDownloaded: false,
    ),
    const LlmModel(
      id: 'phi-3.5-mini',
      name: 'Phi-3.5-mini-3.8B-Instruct',
      category: ModelSizeCategory.medium,
      parameterSize: '3.82B',
      quantization: 'Q4_K_M',
      fileSize: '2.39 GB',
      estimatedRamMb: 2600.0,
      downloadUrl: 'https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF',
      isDownloaded: false,
    ),
    const LlmModel(
      id: 'mistral-7b-v0.3',
      name: 'Mistral-7B-Instruct-v0.3',
      category: ModelSizeCategory.large,
      parameterSize: '7.24B',
      quantization: 'Q4_K_M',
      fileSize: '4.37 GB',
      estimatedRamMb: 4850.0,
      downloadUrl: 'https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF',
      isDownloaded: false,
    ),
    const LlmModel(
      id: 'qwen-7b',
      name: 'Qwen2.5-7B-Instruct',
      category: ModelSizeCategory.large,
      parameterSize: '7.61B',
      quantization: 'Q4_K_M',
      fileSize: '4.68 GB',
      estimatedRamMb: 5120.0,
      downloadUrl: 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF',
      isDownloaded: false,
    ),
  ];
}

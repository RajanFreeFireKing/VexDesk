import 'dart:async';
import 'package:flutter/foundation.dart';
import '../models/llm_model.dart';
import '../bridge/streamvex_bridge.dart';

/// State controller for managing LLM models, download queues, and local imports.
class ModelHubController extends ChangeNotifier {
  final StreamVexBridge _bridge = StreamVexBridge();

  List<LlmModel> _catalog = List.from(LlmModel.defaultCatalog);
  LlmModel? _activeModel;
  bool _isLoading = false;

  List<LlmModel> get catalog => List.unmodifiable(_catalog);
  LlmModel? get activeModel => _activeModel ?? _catalog.firstWhere((m) => m.isLoaded, orElse: () => _catalog.first);
  bool get isLoading => _isLoading;

  List<LlmModel> get smallModels =>
      _catalog.where((m) => m.category == ModelSizeCategory.small).toList();

  List<LlmModel> get mediumModels =>
      _catalog.where((m) => m.category == ModelSizeCategory.medium).toList();

  List<LlmModel> get largeModels =>
      _catalog.where((m) => m.category == ModelSizeCategory.large).toList();

  List<LlmModel> get customModels =>
      _catalog.where((m) => m.category == ModelSizeCategory.custom).toList();

  ModelHubController() {
    _activeModel = _catalog.firstWhere((m) => m.isLoaded, orElse: () => _catalog.first);
    // Initialize bridge with default model
    if (_activeModel != null && _activeModel!.isDownloaded) {
      _bridge.initEngine(_activeModel!.localFilePath ?? _activeModel!.name);
    }
  }

  /// Initiates simulated or real download for a model weights file.
  Future<void> startDownload(String modelId) async {
    final index = _catalog.indexWhere((m) => m.id == modelId);
    if (index == -1) return;

    var model = _catalog[index];
    if (model.isDownloading || model.isDownloaded) return;

    model = model.copyWith(isDownloading: true, downloadProgress: 0.0);
    _catalog[index] = model;
    notifyListeners();

    // Progress increments simulating download stream
    for (int step = 1; step <= 20; step++) {
      await Future.delayed(const Duration(milliseconds: 120));
      final progress = step / 20.0;
      _catalog[index] = _catalog[index].copyWith(downloadProgress: progress);
      notifyListeners();
    }

    _catalog[index] = _catalog[index].copyWith(
      isDownloading: false,
      isDownloaded: true,
      downloadProgress: 1.0,
      localFilePath: 'models/${model.name.toLowerCase().replaceAll(' ', '-')}.gguf',
    );
    notifyListeners();
  }

  /// Loads the selected model into the StreamVex C++ engine resident memory.
  Future<bool> loadModelIntoEngine(String modelId) async {
    _isLoading = true;
    notifyListeners();

    final index = _catalog.indexWhere((m) => m.id == modelId);
    if (index == -1) {
      _isLoading = false;
      notifyListeners();
      return false;
    }

    final targetModel = _catalog[index];
    final success = await _bridge.initEngine(
      targetModel.localFilePath ?? targetModel.name,
    );

    if (success) {
      _catalog = _catalog.map((m) {
        if (m.id == modelId) {
          return m.copyWith(isLoaded: true);
        } else {
          return m.copyWith(isLoaded: false);
        }
      }).toList();
      _activeModel = _catalog[index];
    }

    _isLoading = false;
    notifyListeners();
    return success;
  }

  /// Imports a local .gguf file from device storage (/sdcard or custom path).
  void importLocalModel({
    required String name,
    required String filePath,
    required String parameterSize,
    required String quantization,
    required String fileSize,
    required double estimatedRamMb,
  }) {
    final customModel = LlmModel(
      id: 'custom-${DateTime.now().millisecondsSinceEpoch}',
      name: name,
      category: ModelSizeCategory.custom,
      parameterSize: parameterSize,
      quantization: quantization,
      fileSize: fileSize,
      estimatedRamMb: estimatedRamMb,
      downloadUrl: '',
      localFilePath: filePath,
      isDownloaded: true,
      isDownloading: false,
      downloadProgress: 1.0,
      isLoaded: false,
    );

    _catalog.insert(0, customModel);
    notifyListeners();
  }

  /// Deletes downloaded local weights to reclaim storage.
  void deleteModel(String modelId) {
    final index = _catalog.indexWhere((m) => m.id == modelId);
    if (index == -1) return;

    if (_catalog[index].isLoaded) {
      _bridge.releaseEngine();
    }

    if (_catalog[index].category == ModelSizeCategory.custom) {
      _catalog.removeAt(index);
    } else {
      _catalog[index] = _catalog[index].copyWith(
        isDownloaded: false,
        isLoaded: false,
        downloadProgress: 0.0,
        localFilePath: null,
      );
    }
    notifyListeners();
  }
}

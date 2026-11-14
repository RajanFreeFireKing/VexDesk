import 'package:flutter/material.dart';
import '../../models/llm_model.dart';
import '../../state/model_hub_controller.dart';
import '../../state/chat_controller.dart';
import '../theme/app_theme.dart';

/// Settings & Model Hub Screen.
/// Provides a comprehensive Model Manager with download queues for Small (0.5B),
/// Medium (3B), and Large (7B+) models, plus local file import from storage.
class ModelHubScreen extends StatefulWidget {
  final ModelHubController modelHubController;
  final ChatController chatController;

  const ModelHubScreen({
    super.key,
    required this.modelHubController,
    required this.chatController,
  });

  @override
  State<ModelHubScreen> createState() => _ModelHubScreenState();
}

class _ModelHubScreenState extends State<ModelHubScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  void _showImportLocalFileDialog() {
    final nameCtrl = TextEditingController(text: 'Custom-Llama-Local');
    final pathCtrl = TextEditingController(text: '/sdcard/Download/model-q4_k_m.gguf');
    final paramCtrl = TextEditingController(text: '1.2B');
    final quantCtrl = TextEditingController(text: 'Q4_K_M');
    final ramCtrl = TextEditingController(text: '1200');
    final sizeCtrl = TextEditingController(text: '1.1 GB');

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: AppColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) {
        return Padding(
          padding: EdgeInsets.only(
            left: 20,
            right: 20,
            top: 20,
            bottom: MediaQuery.of(ctx).viewInsets.bottom + 24,
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'Import Local Model File',
                    style: TextStyle(
                      color: AppColors.textPrimary,
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, color: AppColors.textSecondary),
                    onPressed: () => Navigator.of(ctx).pop(),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              const Text(
                'Provide the local file path on device storage (e.g., /sdcard or app directory).',
                style: TextStyle(color: AppColors.textSecondary, fontSize: 13),
              ),
              const SizedBox(height: 16),
              TextField(
                controller: pathCtrl,
                style: const TextStyle(color: AppColors.textPrimary, fontSize: 14),
                decoration: const InputDecoration(
                  labelText: 'Local File Path (.gguf / .bin)',
                  prefixIcon: Icon(Icons.storage, size: 20, color: AppColors.primaryLight),
                ),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: nameCtrl,
                style: const TextStyle(color: AppColors.textPrimary, fontSize: 14),
                decoration: const InputDecoration(
                  labelText: 'Model Display Name',
                  prefixIcon: Icon(Icons.label_outline, size: 20, color: AppColors.primaryLight),
                ),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: paramCtrl,
                      style: const TextStyle(color: AppColors.textPrimary, fontSize: 14),
                      decoration: const InputDecoration(labelText: 'Params (e.g. 1.5B)'),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: TextField(
                      controller: quantCtrl,
                      style: const TextStyle(color: AppColors.textPrimary, fontSize: 14),
                      decoration: const InputDecoration(labelText: 'Quant (e.g. Q4_K_M)'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: ramCtrl,
                      keyboardType: TextInputType.number,
                      style: const TextStyle(color: AppColors.textPrimary, fontSize: 14),
                      decoration: const InputDecoration(labelText: 'Est. RAM (MB)'),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: TextField(
                      controller: sizeCtrl,
                      style: const TextStyle(color: AppColors.textPrimary, fontSize: 14),
                      decoration: const InputDecoration(labelText: 'File Size (e.g. 1.2 GB)'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              SizedBox(
                width: double.infinity,
                height: 48,
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14),
                    ),
                  ),
                  icon: const Icon(Icons.add_circle_outline, color: Colors.white),
                  label: const Text(
                    'Mount & Register Model',
                    style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
                  ),
                  onPressed: () {
                    final ramVal = double.tryParse(ramCtrl.text) ?? 1024.0;
                    widget.modelHubController.importLocalModel(
                      name: nameCtrl.text.trim(),
                      filePath: pathCtrl.text.trim(),
                      parameterSize: paramCtrl.text.trim(),
                      quantization: quantCtrl.text.trim(),
                      fileSize: sizeCtrl.text.trim(),
                      estimatedRamMb: ramVal,
                    );
                    Navigator.of(ctx).pop();
                    _tabController.animateTo(3); // Navigate to Custom tab
                  },
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: widget.modelHubController,
      builder: (context, _) {
        final activeModel = widget.modelHubController.activeModel;

        return Scaffold(
          backgroundColor: AppColors.background,
          appBar: AppBar(
            title: const Text('Model Manager & Downloads'),
            actions: [
              IconButton(
                icon: const Icon(Icons.add_to_photos_rounded),
                tooltip: 'Import Local File',
                onPressed: _showImportLocalFileDialog,
              ),
            ],
            bottom: TabBar(
              controller: _tabController,
              isScrollable: true,
              tabAlignment: TabAlignment.start,
              indicatorColor: AppColors.primaryLight,
              labelColor: AppColors.primaryLight,
              unselectedLabelColor: AppColors.textSecondary,
              tabs: const [
                Tab(text: 'Small (0.5B)'),
                Tab(text: 'Medium (3B)'),
                Tab(text: 'Large (7B+)'),
                Tab(text: 'Custom Local'),
              ],
            ),
          ),
          body: Column(
            children: [
              // Active Engine Memory Allocation Status Banner
              if (activeModel != null)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  decoration: const BoxDecoration(
                    color: AppColors.surface,
                    border: Border(
                      bottom: BorderSide(color: AppColors.borderSubtle, width: 1),
                    ),
                  ),
                  child: Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: AppColors.badgeRamBg,
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: const Icon(Icons.bolt, color: AppColors.emerald, size: 20),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                const Text(
                                  'Resident Engine Model:',
                                  style: TextStyle(
                                    color: AppColors.textSecondary,
                                    fontSize: 12,
                                  ),
                                ),
                                const SizedBox(width: 6),
                                Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                                  decoration: BoxDecoration(
                                    color: AppColors.emerald.withOpacity(0.15),
                                    borderRadius: BorderRadius.circular(4),
                                  ),
                                  child: const Text(
                                    'ACTIVE',
                                    style: TextStyle(
                                      color: AppColors.emerald,
                                      fontSize: 10,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            Text(
                              activeModel.name,
                              style: const TextStyle(
                                color: AppColors.textPrimary,
                                fontSize: 14,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ],
                        ),
                      ),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          const Text('Allocated', style: TextStyle(color: AppColors.textMuted, fontSize: 11)),
                          Text(
                            '~${activeModel.estimatedRamMb.toStringAsFixed(0)} MB',
                            style: const TextStyle(
                              color: AppColors.badgeRamText,
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),

              // Tabbed Model Catalogs
              Expanded(
                child: TabBarView(
                  controller: _tabController,
                  children: [
                    _buildModelList(widget.modelHubController.smallModels, 'small'),
                    _buildModelList(widget.modelHubController.mediumModels, 'medium'),
                    _buildModelList(widget.modelHubController.largeModels, 'large'),
                    _buildModelList(widget.modelHubController.customModels, 'custom'),
                  ],
                ),
              ),
            ],
          ),
          floatingActionButton: FloatingActionButton.extended(
            backgroundColor: AppColors.primary,
            icon: const Icon(Icons.file_upload_outlined, color: Colors.white),
            label: const Text(
              'Import Local File',
              style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
            ),
            onPressed: _showImportLocalFileDialog,
          ),
        );
      },
    );
  }

  Widget _buildModelList(List<LlmModel> models, String categoryKey) {
    if (models.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.folder_off_outlined, size: 48, color: AppColors.textMuted),
              const SizedBox(height: 12),
              Text(
                categoryKey == 'custom'
                    ? 'No local files imported yet'
                    : 'No models found in this tier',
                style: const TextStyle(color: AppColors.textSecondary, fontSize: 15),
              ),
              const SizedBox(height: 6),
              const Text(
                'Tap "Import Local File" to load on-device weights from storage.',
                textAlign: TextAlign.center,
                style: TextStyle(color: AppColors.textMuted, fontSize: 13),
              ),
            ],
          ),
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: models.length,
      itemBuilder: (context, index) {
        return _ModelCard(
          model: models[index],
          onDownload: () => widget.modelHubController.startDownload(models[index].id),
          onLoad: () async {
            final ok = await widget.modelHubController.loadModelIntoEngine(models[index].id);
            if (ok) {
              widget.chatController.updateActiveModel(models[index]);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text('Loaded ${models[index].name} into StreamVex C++ engine'),
                  behavior: SnackBarBehavior.floating,
                  backgroundColor: AppColors.surfaceElevated,
                ),
              );
            }
          },
          onDelete: () => widget.modelHubController.deleteModel(models[index].id),
        );
      },
    );
  }
}

class _ModelCard extends StatelessWidget {
  final LlmModel model;
  final VoidCallback onDownload;
  final VoidCallback onLoad;
  final VoidCallback onDelete;

  const _ModelCard({
    required this.model,
    required this.onDownload,
    required this.onLoad,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surfaceElevated,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: model.isLoaded ? AppColors.emerald.withOpacity(0.5) : AppColors.borderSubtle,
          width: model.isLoaded ? 1.5 : 1.0,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          model.name,
                          style: const TextStyle(
                            color: AppColors.textPrimary,
                            fontSize: 15.5,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        if (model.isLoaded) ...[
                          const SizedBox(width: 8),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                              color: AppColors.emerald.withOpacity(0.2),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: const Text(
                              'LOADED',
                              style: TextStyle(
                                color: AppColors.emerald,
                                fontSize: 10,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ],
                      ],
                    ),
                    const SizedBox(height: 6),
                    Wrap(
                      spacing: 8,
                      children: [
                        _specChip(Icons.compress, model.quantization),
                        _specChip(Icons.straighten, model.parameterSize),
                        _specChip(Icons.sd_card_outlined, model.fileSize),
                        _specChip(Icons.memory, '~${model.estimatedRamMb.toStringAsFixed(0)} MB RAM'),
                      ],
                    ),
                  ],
                ),
              ),
              if (model.isDownloaded)
                IconButton(
                  icon: const Icon(Icons.delete_outline, color: AppColors.rose, size: 20),
                  tooltip: 'Delete weights',
                  onPressed: onDelete,
                ),
            ],
          ),

          if (model.isDownloading) ...[
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(
                value: model.downloadProgress,
                minHeight: 6,
                backgroundColor: AppColors.surface,
                valueColor: const AlwaysStoppedAnimation(AppColors.secondary),
              ),
            ),
            const SizedBox(height: 4),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Downloading model weights...', style: TextStyle(color: AppColors.textMuted, fontSize: 11)),
                Text(
                  '${(model.downloadProgress * 100).toStringAsFixed(0)}%',
                  style: const TextStyle(
                    color: AppColors.secondary,
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ],

          const SizedBox(height: 14),
          Row(
            children: [
              if (!model.isDownloaded && !model.isDownloading)
                Expanded(
                  child: OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      foregroundColor: AppColors.secondary,
                      side: const BorderSide(color: AppColors.secondary),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                    icon: const Icon(Icons.download_rounded, size: 18),
                    label: const Text('Download Weights'),
                    onPressed: onDownload,
                  ),
                ),
              if (model.isDownloaded)
                Expanded(
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: model.isLoaded ? AppColors.surfaceHighlight : AppColors.primary,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                    icon: Icon(
                      model.isLoaded ? Icons.check_circle_outline : Icons.play_arrow_rounded,
                      size: 18,
                    ),
                    label: Text(model.isLoaded ? 'Currently Active' : 'Load Into StreamVex Engine'),
                    onPressed: model.isLoaded ? null : onLoad,
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _specChip(IconData icon, String label) {
    return Container(
      margin: const EdgeInsets.only(top: 4),
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: AppColors.borderSubtle),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: AppColors.textSecondary),
          const SizedBox(width: 4),
          Text(
            label,
            style: const TextStyle(color: AppColors.textSecondary, fontSize: 11.5),
          ),
        ],
      ),
    );
  }
}

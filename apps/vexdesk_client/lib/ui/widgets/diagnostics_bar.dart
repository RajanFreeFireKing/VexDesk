import 'package:flutter/material.dart';
import '../../models/engine_stats.dart';
import '../theme/app_theme.dart';

/// Top diagnostics bar displaying live engine metrics:
/// - Real-time tokens/sec
/// - Current RAM consumption (MB)
/// - Active model name
/// - Engine bridge status (Native FFI vs Simulator)
class DiagnosticsBar extends StatelessWidget {
  final EngineStats stats;
  final VoidCallback? onTapModel;

  const DiagnosticsBar({
    super.key,
    required this.stats,
    this.onTapModel,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: AppColors.surface,
        border: const Border(
          bottom: BorderSide(color: AppColors.borderSubtle, width: 1),
        ),
      ),
      child: Row(
        children: [
          // Active Model Chip
          Expanded(
            child: InkWell(
              onTap: onTapModel,
              borderRadius: BorderRadius.circular(8),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: AppColors.badgeModelBg,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: AppColors.primary.withOpacity(0.3)),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.memory_rounded,
                      size: 15,
                      color: AppColors.badgeModelText,
                    ),
                    const SizedBox(width: 6),
                    Flexible(
                      child: Text(
                        stats.modelName,
                        style: const TextStyle(
                          color: AppColors.badgeModelText,
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ),
                    const SizedBox(width: 4),
                    const Icon(
                      Icons.arrow_drop_down,
                      size: 16,
                      color: AppColors.badgeModelText,
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),

          // Live Tokens / Sec Badge
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
            decoration: BoxDecoration(
              color: AppColors.badgeTpsBg,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: AppColors.secondary.withOpacity(0.3)),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _StatusPulseDot(isActive: stats.isGenerating),
                const SizedBox(width: 6),
                Text(
                  stats.isGenerating
                      ? '${stats.tokensPerSec.toStringAsFixed(1)} t/s'
                      : (stats.tokensPerSec > 0
                          ? '${stats.tokensPerSec.toStringAsFixed(1)} t/s'
                          : '-- t/s'),
                  style: const TextStyle(
                    color: AppColors.badgeTpsText,
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    fontFeatures: [FontFeature.tabularFigures()],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 8),

          // RAM Consumption (MB) Badge
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
            decoration: BoxDecoration(
              color: AppColors.badgeRamBg,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: AppColors.emerald.withOpacity(0.3)),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(
                  Icons.developer_board,
                  size: 14,
                  color: AppColors.badgeRamText,
                ),
                const SizedBox(width: 5),
                Text(
                  '${stats.ramUsageMb.toStringAsFixed(0)} MB',
                  style: const TextStyle(
                    color: AppColors.badgeRamText,
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    fontFeatures: [FontFeature.tabularFigures()],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _StatusPulseDot extends StatefulWidget {
  final bool isActive;
  const _StatusPulseDot({required this.isActive});

  @override
  State<_StatusPulseDot> createState() => _StatusPulseDotState();
}

class _StatusPulseDotState extends State<_StatusPulseDot>
    with SingleTickerProviderStateMixin {
  late AnimationController _animController;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.isActive) {
      return Container(
        width: 7,
        height: 7,
        decoration: const BoxDecoration(
          color: AppColors.textMuted,
          shape: BoxShape.circle,
        ),
      );
    }

    return FadeTransition(
      opacity: _animController,
      child: Container(
        width: 7,
        height: 7,
        decoration: const BoxDecoration(
          color: AppColors.secondary,
          shape: BoxShape.circle,
        ),
      ),
    );
  }
}

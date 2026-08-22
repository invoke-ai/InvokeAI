import type { VideoWidgetValues } from '@features/video/core/types';

/**
 * Content comparators for the widget's memo boundaries and the mount-time
 * reconciler. The widget re-derives `values` on every patch, so equality by
 * content — not identity — decides whether a write-back or re-render is due.
 */

const stableStringify = (value: unknown): string => JSON.stringify(value ?? null);

export const areVideoModelsEquivalent = (
  left: { key: string; hash?: string } | null,
  right: { key: string; hash?: string } | null
): boolean => (left === null && right === null) || (left?.key === right?.key && left?.hash === right?.hash);

export const areVideoLorasEquivalent = (left: VideoWidgetValues['loras'], right: VideoWidgetValues['loras']): boolean =>
  left.length === right.length &&
  left.every(
    (lora, index) =>
      lora.model.key === right[index]?.model.key &&
      lora.weight === right[index]?.weight &&
      lora.isEnabled === right[index]?.isEnabled
  );

export const areVideoValuesEqual = (left: VideoWidgetValues, right: VideoWidgetValues): boolean => {
  if (left === right) {
    return true;
  }

  return (
    left.modelKey === right.modelKey &&
    areVideoModelsEquivalent(left.model, right.model) &&
    left.positivePrompt === right.positivePrompt &&
    left.negativePrompt === right.negativePrompt &&
    left.negativePromptEnabled === right.negativePromptEnabled &&
    left.positivePromptHeightPx === right.positivePromptHeightPx &&
    left.negativePromptHeightPx === right.negativePromptHeightPx &&
    left.aspectRatioId === right.aspectRatioId &&
    left.targetResolution === right.targetResolution &&
    left.numFrames === right.numFrames &&
    left.fps === right.fps &&
    left.steps === right.steps &&
    left.cfgScale === right.cfgScale &&
    left.cfgScaleLowNoise === right.cfgScaleLowNoise &&
    left.acceleratorEnabled === right.acceleratorEnabled &&
    stableStringify(left.acceleratorLoraKeys) === stableStringify(right.acceleratorLoraKeys) &&
    left.seed === right.seed &&
    left.shouldRandomizeSeed === right.shouldRandomizeSeed &&
    left.batchCount === right.batchCount &&
    areVideoLorasEquivalent(left.loras, right.loras) &&
    stableStringify(left.firstFrameImage) === stableStringify(right.firstFrameImage) &&
    stableStringify(left.lastFrameImage) === stableStringify(right.lastFrameImage) &&
    stableStringify(left.sourceVideo) === stableStringify(right.sourceVideo) &&
    stableStringify(left.vae) === stableStringify(right.vae) &&
    stableStringify(left.wanT5EncoderModel) === stableStringify(right.wanT5EncoderModel) &&
    stableStringify(left.wanLowNoiseModel) === stableStringify(right.wanLowNoiseModel) &&
    stableStringify(left.componentSourceModel) === stableStringify(right.componentSourceModel) &&
    stableStringify(left.h3TransformerModel) === stableStringify(right.h3TransformerModel) &&
    stableStringify(left.h3TextEncoderModel) === stableStringify(right.h3TextEncoderModel)
  );
};

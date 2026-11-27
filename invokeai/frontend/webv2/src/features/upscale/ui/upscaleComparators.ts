import type { GenerateLora } from '@features/generation/contracts';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';

/** Compare normalized values by content; permissive memo comparisons would silently retain stale UI. */

export const areStringArraysEqual = (left: readonly string[], right: readonly string[]): boolean =>
  left === right || (left.length === right.length && left.every((value, index) => value === right[index]));

export const areLorasEquivalent = (left: readonly GenerateLora[], right: readonly GenerateLora[]): boolean =>
  left.length === right.length &&
  left.every((lora, index) => {
    const other = right[index];

    return (
      other !== undefined &&
      lora.model.key === other.model.key &&
      lora.isEnabled === other.isEnabled &&
      lora.weight === other.weight
    );
  });

export const getModelTriggerPhrases = (model: UpscaleWidgetValues['model']): readonly string[] => {
  const phrases = (model as { trigger_phrases?: unknown } | null)?.trigger_phrases;

  return Array.isArray(phrases) ? phrases.filter((phrase): phrase is string => typeof phrase === 'string') : [];
};

/**
 * Catalog refreshes can change trigger phrases or base under the same key; compare prompt-relevant metadata after
 * identity checks.
 */
export const areModelsEquivalent = (left: UpscaleWidgetValues['model'], right: UpscaleWidgetValues['model']): boolean =>
  left === right ||
  (left !== null &&
    right !== null &&
    left.key === right.key &&
    left.base === right.base &&
    left.name === right.name &&
    areStringArraysEqual(getModelTriggerPhrases(left), getModelTriggerPhrases(right)));

export const areInputImagesEquivalent = (
  left: UpscaleWidgetValues['inputImage'],
  right: UpscaleWidgetValues['inputImage']
): boolean =>
  left === right ||
  (left !== null &&
    right !== null &&
    left.image_name === right.image_name &&
    left.width === right.width &&
    left.height === right.height);

/** Use structural equality to detect any normalization change at the reconciler boundary. */
export const valuesAreEqual = (left: UpscaleWidgetValues, right: UpscaleWidgetValues): boolean =>
  JSON.stringify(left) === JSON.stringify(right);

import type { GenerateLora } from '@features/generation/contracts';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';

/**
 * Content comparators for the Upscale widget's memo boundaries.
 *
 * `values` is re-derived from the raw widget state on every patch. Its nested
 * members are mostly identity-stable — `normalizeUpscaleWidgetValues` preserves
 * them across a patch that didn't touch them — but that's an incidental
 * property of the normalizer, not a contract the widget should lean on. These
 * compare by content instead, so no section depends on that incidental
 * stability to avoid re-rendering (e.g. the prompt editors on a scale edit).
 *
 * They live apart from the components that pass them to `memo` because that is
 * the only way they can be tested directly: a comparator that is wrong in the
 * conservative direction merely re-renders, while one that is wrong in the
 * permissive direction silently shows stale data, and only the second is a bug
 * you can miss by looking at the screen.
 */

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
 * `key` alone is not enough: a catalog refresh swaps in a fresh config under
 * the same key, and the prompt editors consume per-config data (trigger
 * phrases, used both to populate the autocomplete and to label its group;
 * `base`, used to filter compatible embeddings). Identity short-circuits the
 * common case.
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

/**
 * Whole-values equality for the model reconciler, which asks only "did
 * normalization change anything at all" — a question a deep structural compare
 * answers correctly and cheaply enough at that one call site.
 */
export const valuesAreEqual = (left: UpscaleWidgetValues, right: UpscaleWidgetValues): boolean =>
  JSON.stringify(left) === JSON.stringify(right);

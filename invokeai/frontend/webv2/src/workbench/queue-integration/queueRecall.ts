import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { QueueGenerationMeta } from '@features/queue/contracts';
import type { VideoWidgetValues } from '@features/video';
import type { ImageRecallCapabilities, ImageRecallKind } from '@workbench/image-actions';

import { cloneVideoWidgetValues } from '@features/video';

/**
 * Partial recall merges current values; all/remix restores the submission snapshot. Foreign items expose only
 * session prompts and executed seed.
 */

export const getQueueRecallCapabilities = (
  snapshot: GenerateWidgetValues | null,
  meta: QueueGenerationMeta
): ImageRecallCapabilities => ({
  all: snapshot !== null,
  clipSkip: snapshot !== null,
  dimensions: snapshot !== null,
  prompts: snapshot !== null || meta.positivePrompt !== undefined,
  remix: snapshot !== null,
  seed: meta.seed !== undefined || (snapshot !== null && snapshot.seedMode !== 'random'),
  workflow: false,
});

export const getVideoQueueRecallCapabilities = (
  snapshot: VideoWidgetValues | null,
  meta: QueueGenerationMeta
): ImageRecallCapabilities => ({
  all: snapshot !== null,
  clipSkip: false,
  dimensions: false,
  prompts: snapshot !== null || meta.positivePrompt !== undefined,
  remix: snapshot !== null,
  seed: meta.seed !== undefined || (snapshot !== null && snapshot.seedMode !== 'random'),
  workflow: false,
});

/** The prompt tool model picks are the widget's preference, not part of what was submitted. */
const withCurrentToolModelPicks = (
  values: GenerateWidgetValues,
  current: GenerateWidgetValues | null
): GenerateWidgetValues =>
  current
    ? {
        ...values,
        expandPromptModelKey: current.expandPromptModelKey ?? null,
        imageToPromptModelKey: current.imageToPromptModelKey ?? null,
      }
    : values;

export const buildQueueRecallValues = (
  kind: ImageRecallKind,
  {
    current,
    meta,
    snapshot,
  }: {
    current: GenerateWidgetValues | null;
    meta: QueueGenerationMeta;
    snapshot: GenerateWidgetValues | null;
  }
): GenerateWidgetValues | null => {
  if (kind === 'all') {
    return snapshot ? withCurrentToolModelPicks(snapshot, current) : null;
  }

  if (kind === 'remix') {
    return snapshot ? withCurrentToolModelPicks({ ...snapshot, seedMode: 'random' }, current) : null;
  }

  if (!current) {
    return null;
  }

  if (kind === 'prompts') {
    const positivePrompt = snapshot?.positivePrompt ?? meta.positivePrompt;

    if (positivePrompt === undefined) {
      return null;
    }

    const negativePrompt = snapshot?.negativePrompt ?? meta.negativePrompt;

    return {
      ...current,
      positivePrompt,
      // Restore templates with authored snapshot prompts; clear templates for already-merged session prompts to
      // avoid double wrapping. Older snapshots must not write undefined templates.
      promptTemplate: snapshot ? (snapshot.promptTemplate ?? null) : null,
      ...(negativePrompt !== undefined
        ? { negativePrompt, negativePromptEnabled: snapshot?.negativePromptEnabled ?? negativePrompt.length > 0 }
        : {}),
    };
  }

  if (kind === 'seed') {
    // The session's seed is what actually ran (randomized submissions store a
    // placeholder in the snapshot), so it wins.
    const seed = meta.seed ?? (snapshot && snapshot.seedMode !== 'random' ? snapshot.seed : undefined);

    return seed === undefined ? null : { ...current, seed, seedMode: 'fixed' };
  }

  if (kind === 'dimensions') {
    return snapshot
      ? {
          ...current,
          aspectRatioId: snapshot.aspectRatioId,
          aspectRatioIsLocked: snapshot.aspectRatioIsLocked,
          aspectRatioValue: snapshot.aspectRatioValue,
          height: snapshot.height,
          width: snapshot.width,
        }
      : null;
  }

  return snapshot ? { ...current, clipSkip: snapshot.clipSkip } : null;
};

export const buildVideoQueueRecallPatch = (
  kind: ImageRecallKind,
  meta: QueueGenerationMeta,
  snapshot: VideoWidgetValues | null = null
): Partial<VideoWidgetValues> | null => {
  if (kind === 'all' || kind === 'remix') {
    if (!snapshot) {
      return null;
    }
    const values = cloneVideoWidgetValues(snapshot);
    return kind === 'remix' ? { ...values, seedMode: 'random' } : values;
  }
  if (kind === 'prompts') {
    if (snapshot) {
      return {
        positivePrompt: snapshot.positivePrompt,
        negativePrompt: snapshot.negativePrompt,
        negativePromptEnabled: snapshot.negativePromptEnabled,
      };
    }
    if (meta.positivePrompt === undefined) {
      return null;
    }

    return {
      positivePrompt: meta.positivePrompt,
      // Absent/empty session negatives cannot identify disabled state, so preserve the toggle. H3 batch negatives
      // may recall harmless hidden values absent from gallery metadata.
      ...(meta.negativePrompt !== undefined && meta.negativePrompt.length > 0
        ? { negativePrompt: meta.negativePrompt, negativePromptEnabled: true }
        : {}),
    };
  }

  if (kind === 'seed') {
    const seed = meta.seed ?? (snapshot && snapshot.seedMode !== 'random' ? snapshot.seed : undefined);
    return seed === undefined ? null : { seed, seedMode: 'fixed' };
  }

  return null;
};

/**
 * Discriminate Video patches from complete Generate values so callers must handle the correct destination
 * contract.
 */
export type QueueRecallPlan =
  | { target: 'generate'; values: GenerateWidgetValues }
  | { target: 'video'; patch: Partial<VideoWidgetValues> };

export const planQueueRecall = (
  kind: ImageRecallKind,
  {
    current,
    isVideoItem,
    meta,
    snapshot,
    videoSnapshot = null,
  }: {
    current: GenerateWidgetValues | null;
    /**
     * Whether this item was submitted from Video BY THIS CLIENT. False also
     * means "unknown" — a foreign item's `field_values` cannot distinguish a
     * video batch from an image one, so the Generate-shaped default stands.
     */
    isVideoItem: boolean;
    meta: QueueGenerationMeta;
    snapshot: GenerateWidgetValues | null;
    videoSnapshot?: VideoWidgetValues | null;
  }
): QueueRecallPlan | null => {
  if (isVideoItem) {
    const patch = buildVideoQueueRecallPatch(kind, meta, videoSnapshot);

    return patch ? { patch, target: 'video' } : null;
  }

  const values = buildQueueRecallValues(kind, { current, meta, snapshot });

  return values ? { target: 'generate', values } : null;
};

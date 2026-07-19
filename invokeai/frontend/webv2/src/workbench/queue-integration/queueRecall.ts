import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { QueueGenerationMeta } from '@features/queue/contracts';
import type { ImageRecallCapabilities, ImageRecallKind } from '@workbench/image-actions';

/**
 * Snapshot/session-based recall for queue items, mirroring the image-metadata
 * recall semantics: partial kinds merge the recalled fields into the CURRENT
 * generate form values; `all`/`remix` replace them with the submission
 * snapshot. Items this client submitted carry the full snapshot; foreign items
 * still expose prompts + the actual executed seed via the session meta.
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
  seed: meta.seed !== undefined || (snapshot !== null && !snapshot.shouldRandomizeSeed),
});

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
    return snapshot;
  }

  if (kind === 'remix') {
    return snapshot ? { ...snapshot, shouldRandomizeSeed: true } : null;
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
      ...(negativePrompt !== undefined
        ? { negativePrompt, negativePromptEnabled: snapshot?.negativePromptEnabled ?? negativePrompt.length > 0 }
        : {}),
    };
  }

  if (kind === 'seed') {
    // The session's seed is what actually ran (randomized submissions store a
    // placeholder in the snapshot), so it wins.
    const seed = meta.seed ?? (snapshot && !snapshot.shouldRandomizeSeed ? snapshot.seed : undefined);

    return seed === undefined ? null : { ...current, seed, shouldRandomizeSeed: false };
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

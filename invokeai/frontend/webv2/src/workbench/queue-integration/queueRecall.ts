import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { QueueGenerationMeta, QueueItemReadModel } from '@features/queue/contracts';
import type { VideoWidgetValues } from '@features/video';
import type { ImageRecallCapabilities, ImageRecallKind } from '@workbench/image-actions';

import { extractGenerationMeta, extractNamedGenerationMeta, isWebv2QueueItemOrigin } from '@features/queue/contracts';
import { cloneVideoWidgetValues } from '@features/video';

/**
 * Partial recall merges current values; all/remix restores the submission snapshot. Foreign items expose only session
 * prompts and executed seed.
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

/**
 * Whether the Generate panel can take a partial recall: one merges into its current values, which exist once a
 * supported model is selected under a loaded capability table. `loading` waits for the model catalog to load or fail:
 * it can hold the model the panel will select.
 */
export type QueueRecallTargetState = 'ready' | 'loading' | 'no-model' | 'no-capabilities';

export const getQueueRecallTargetState = ({
  hasCapabilityTable,
  hasCurrentValues,
  isCatalogSettled,
}: {
  hasCapabilityTable: boolean;
  hasCurrentValues: boolean;
  isCatalogSettled: boolean;
}): QueueRecallTargetState =>
  hasCurrentValues ? 'ready' : !hasCapabilityTable ? 'no-capabilities' : isCatalogSettled ? 'no-model' : 'loading';

/**
 * What submitted an item, as far as this browser can tell. Its snapshot names the source; Workflow and Upscale
 * snapshots record no settings. Without one, `readQueueItemRecallInput` says whether the item ran something other than
 * a Generate-style graph. `unknown` is an item another browser or client queued, which may have come from Generate.
 */
export type QueueRecallSource = 'generate' | 'video' | 'other' | 'unknown';

export const getQueueRecallSource = (snapshotSourceId: string | null, isNonGenerateRun: boolean): QueueRecallSource =>
  snapshotSourceId === 'generate' || snapshotSourceId === 'canvas'
    ? 'generate'
    : snapshotSourceId === 'video'
      ? 'video'
      : snapshotSourceId !== null || isNonGenerateRun
        ? 'other'
        : 'unknown';

/**
 * What a queue item's session tells recall. The server keeps a submitted workflow, named or not. webv2's own Generate,
 * Canvas, Video and Upscale submissions name their prompts and seed, so a webv2 item without them ran something else,
 * and its other values are never read by type and order; items from other clients still are.
 */
export const readQueueItemRecallInput = (
  item: QueueItemReadModel
): { isNonGenerateRun: boolean; meta: QueueGenerationMeta } => {
  const ranWorkflow = typeof item.workflowName === 'string';

  if (!isWebv2QueueItemOrigin(item.origin)) {
    return { isNonGenerateRun: ranWorkflow, meta: extractGenerationMeta(item) };
  }

  const meta = extractNamedGenerationMeta(item);

  return { isNonGenerateRun: ranWorkflow || (meta.positivePrompt === undefined && meta.seed === undefined), meta };
};

export interface QueueRecallAvailability {
  capabilities: ImageRecallCapabilities;
  /** For each unavailable kind, the i18n key of why; none while the answer is still loading. */
  reasons: Partial<Record<ImageRecallKind, string>>;
}

const RECALL_KINDS = ['all', 'remix', 'prompts', 'seed', 'dimensions', 'clipSkip'] as const satisfies ImageRecallKind[];

const EMPTY_CAPABILITIES: ImageRecallCapabilities = {
  all: false,
  clipSkip: false,
  dimensions: false,
  prompts: false,
  remix: false,
  seed: false,
  workflow: false,
};

const getUnrecordedReason = (kind: ImageRecallKind, source: QueueRecallSource, hasSnapshot: boolean): string =>
  source === 'video' && (kind === 'dimensions' || kind === 'clipSkip')
    ? 'widgets.queue.recallNotForVideo'
    : hasSnapshot
      ? // With a snapshot only a random seed is ever missing: the server has not reported the one it chose.
        'widgets.queue.recallRandomSeed'
      : 'widgets.queue.recallSettingsNotKept';

/**
 * What each recall kind can do for an item, and why not when it cannot; every reason holds wherever it is shown.
 * Without this browser's snapshot only the prompts and seed the server ran are known, and a Workflow or Upscale run
 * holds no Generate settings at all (its session values are not Generate's, whatever their names). A missing record
 * outranks an unready Generate panel: selecting a model would not make it recallable.
 */
export const getQueueRecallAvailability = ({
  meta,
  snapshot,
  source,
  target,
  videoSnapshot,
}: {
  meta: QueueGenerationMeta;
  snapshot: GenerateWidgetValues | null;
  source: QueueRecallSource;
  target: QueueRecallTargetState;
  videoSnapshot: VideoWidgetValues | null;
}): QueueRecallAvailability => {
  const reasons: QueueRecallAvailability['reasons'] = {};

  if (source === 'other') {
    for (const kind of RECALL_KINDS) {
      reasons[kind] = 'widgets.queue.recallUnavailableForItem';
    }
    return { capabilities: EMPTY_CAPABILITIES, reasons };
  }

  const isVideo = source === 'video';
  const hasSnapshot = isVideo ? videoSnapshot !== null : snapshot !== null;
  const recorded = isVideo
    ? getVideoQueueRecallCapabilities(videoSnapshot, meta)
    : getQueueRecallCapabilities(snapshot, meta);
  const capabilities = { ...recorded };
  // Partial kinds then wait for the Generate panel while nothing else is recallable, which the "Nothing recallable"
  // reasons say; beside an available full recall they name only what the panel lacks.
  const blockedReason =
    target === 'no-capabilities'
      ? hasSnapshot
        ? 'widgets.generate.capabilitiesUnavailableForSetup'
        : 'widgets.queue.recallUnavailableCapabilities'
      : target === 'no-model'
        ? hasSnapshot
          ? 'widgets.generate.selectSupportedModelFirst'
          : 'widgets.queue.recallUnavailable'
        : null;

  for (const kind of RECALL_KINDS) {
    if (!recorded[kind]) {
      reasons[kind] = getUnrecordedReason(kind, source, hasSnapshot);
    } else if (!isVideo && kind !== 'all' && kind !== 'remix' && target !== 'ready') {
      capabilities[kind] = false;
      if (blockedReason) {
        reasons[kind] = blockedReason;
      }
    }
  }

  return { capabilities, reasons };
};

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
      // Restore templates with authored snapshot prompts; clear templates for already-merged session prompts to avoid
      // double wrapping. Older snapshots must not write undefined templates.
      promptTemplate: snapshot ? (snapshot.promptTemplate ?? null) : null,
      ...(negativePrompt !== undefined
        ? { negativePrompt, negativePromptEnabled: snapshot?.negativePromptEnabled ?? negativePrompt.length > 0 }
        : {}),
    };
  }

  if (kind === 'seed') {
    // The session's seed is what actually ran (randomized submissions store a placeholder in the snapshot), so it wins.
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
      // Absent/empty session negatives cannot identify disabled state, so preserve the toggle. H3 batch negatives may
      // recall harmless hidden values absent from gallery metadata.
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

/** Discriminate Video patches from complete Generate values so callers must handle the correct destination contract. */
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
     * Whether this item was submitted from Video BY THIS CLIENT. False also means "unknown" — a foreign item's
     * `field_values` cannot distinguish a video batch from an image one, so the Generate-shaped default stands.
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

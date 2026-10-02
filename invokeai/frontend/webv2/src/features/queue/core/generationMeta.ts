import type { QueueItemReadModel } from '@features/queue/core/types';

/**
 * Prefer local snapshots, then named node fields. Preserve blank positive prompts; only unknown graph fields fall
 * back to type/order heuristics.
 */

export interface QueueGenerationMeta {
  positivePrompt?: string;
  negativePrompt?: string;
  seed?: number;
}

/** The node ids the generate/video graphs give their prompt and seed nodes. */
const POSITIVE_PROMPT_NODE_PATH = 'positive_prompt';
const NEGATIVE_PROMPT_NODE_PATH = 'negative_prompt';
const SEED_NODE_PATH = 'seed';

export const extractGenerationMeta = (item: QueueItemReadModel): QueueGenerationMeta => {
  const fieldValues = item.fieldValues ?? [];
  const meta: QueueGenerationMeta = {};
  let namedPrompt = false;

  for (const { nodePath, value } of fieldValues) {
    if (nodePath === POSITIVE_PROMPT_NODE_PATH && typeof value === 'string') {
      // Recorded verbatim, empty included: an empty positive is a real
      // submission, not a missing field.
      meta.positivePrompt = value;
      namedPrompt = true;
    } else if (nodePath === NEGATIVE_PROMPT_NODE_PATH && typeof value === 'string') {
      meta.negativePrompt = value;
      namedPrompt = true;
    } else if (nodePath === SEED_NODE_PATH && typeof value === 'number') {
      meta.seed = value;
    }
  }

  if (namedPrompt && meta.seed !== undefined) {
    return meta;
  }

  // Fill unresolved metadata by type/order only for graphs with unfamiliar node names.
  const prompts: string[] = [];

  for (const { value } of fieldValues) {
    if (typeof value === 'number' && meta.seed === undefined) {
      meta.seed = value;
    } else if (typeof value === 'string' && value.trim().length > 0) {
      prompts.push(value);
    }
  }

  if (namedPrompt) {
    return meta;
  }

  if (prompts[0] !== undefined) {
    meta.positivePrompt = prompts[0];
  }

  if (prompts[1] !== undefined) {
    meta.negativePrompt = prompts[1];
  }

  return meta;
};

/** First result image name from a completed item's session, or null. */
export const getResultImageName = (item: QueueItemReadModel): string | null => {
  return item.resultImageNames[0] ?? null;
};

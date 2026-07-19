import type { QueueItemReadModel } from '@features/queue/core/types';

/**
 * Best-effort recovery of the human-facing generation parameters for a queue
 * item. The reliable source is the local submission snapshot (matched in the
 * view by origin), but server items from other clients/sources only carry
 * field values — generic node/field/value substitutions. For
 * webv2 generate batches these are exactly the seed (number) + positive/negative
 * prompts (strings), so we recover them by value type and order, degrading
 * gracefully for unknown graphs.
 */

export interface QueueGenerationMeta {
  positivePrompt?: string;
  negativePrompt?: string;
  seed?: number;
}

export const extractGenerationMeta = (item: QueueItemReadModel): QueueGenerationMeta => {
  const meta: QueueGenerationMeta = {};
  const prompts: string[] = [];

  for (const fieldValue of item.fieldValues ?? []) {
    const { value } = fieldValue;

    if (typeof value === 'number' && meta.seed === undefined) {
      meta.seed = value;
    } else if (typeof value === 'string' && value.trim().length > 0) {
      prompts.push(value);
    }
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

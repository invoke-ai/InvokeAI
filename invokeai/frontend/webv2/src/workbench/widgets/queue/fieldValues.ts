import type { QueueServerItem } from './queueServerApi';

/**
 * Best-effort recovery of the human-facing generation parameters for a queue
 * item. The reliable source is the local submission snapshot (matched in the
 * view by origin), but server items from other clients/sources only carry
 * `field_values` — generic `{node_path, field_name, value}` substitutions. For
 * webv2 generate batches these are exactly the seed (number) + positive/negative
 * prompts (strings), so we recover them by value type and order, degrading
 * gracefully for unknown graphs.
 */

export interface QueueGenerationMeta {
  positivePrompt?: string;
  negativePrompt?: string;
  seed?: number;
}

export const extractGenerationMeta = (item: QueueServerItem): QueueGenerationMeta => {
  const meta: QueueGenerationMeta = {};
  const prompts: string[] = [];

  for (const fieldValue of item.field_values ?? []) {
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
export const getResultImageName = (item: QueueServerItem): string | null => {
  for (const result of Object.values(item.session?.results ?? {})) {
    if (!result || typeof result !== 'object') {
      continue;
    }

    const image = (result as { image?: { image_name?: unknown } }).image;
    if (typeof image?.image_name === 'string') {
      return image.image_name;
    }

    const collection = (result as { collection?: unknown }).collection;
    if (Array.isArray(collection)) {
      for (const entry of collection) {
        const imageName = entry && typeof entry === 'object' ? (entry as { image_name?: unknown }).image_name : null;
        if (typeof imageName === 'string') {
          return imageName;
        }
      }
    }
  }

  return null;
};

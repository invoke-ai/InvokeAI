import type { WanLoraTarget } from '@features/generation/contracts';

import { isWanLoraTarget } from '@features/generation/settings';

/** Every field optional on read; pre-identifier records name `{ model_name, base_model, model_type }`. */
export interface RecordedModelRef {
  base: string | null;
  hash: string | null;
  key: string | null;
  name: string | null;
  type: string | null;
}

interface ResolvableModel {
  base: string;
  hash?: string;
  /** Registered, but its files are gone. */
  is_missing?: boolean;
  key: string;
  name: string;
  type: string;
}

/**
 * `hashMiss: 'refuse'` restores only the weights that ran; `'matchName'` falls back to name and base for reinstalls.
 * `ambiguous: 'refuse'` reports several matches; `'first'` takes catalog order.
 */
export interface RecordedModelPolicy {
  ambiguous: 'first' | 'refuse';
  hashMiss: 'matchName' | 'refuse';
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const readString = (record: Record<string, unknown>, key: string): string | null => {
  const value = record[key];

  return typeof value === 'string' ? value : null;
};

/** Null when not a record at all. Display reads this; recall needs `toRecordedModelRef`. */
export const readRecordedModelRef = (value: unknown): RecordedModelRef | null =>
  isRecord(value)
    ? {
        base: readString(value, 'base') ?? readString(value, 'base_model'),
        hash: readString(value, 'hash'),
        key: readString(value, 'key'),
        name: readString(value, 'name') ?? readString(value, 'model_name'),
        type: readString(value, 'type') ?? readString(value, 'model_type'),
      }
    : null;

/** Null unless resolution can act on it: an install-local key, a content hash, or name+base. */
export const toRecordedModelRef = (value: unknown): RecordedModelRef | null => {
  const ref = readRecordedModelRef(value);

  return ref && (ref.key || ref.hash || (ref.name && ref.base)) ? ref : null;
};

/** The content hash when recorded, else name and base (and type when recorded). */
const matchRecordedModel = <T extends ResolvableModel>(
  ref: RecordedModelRef,
  models: readonly T[],
  policy: RecordedModelPolicy
): T | 'ambiguous' | null => {
  const byHash = ref.hash ? models.filter((model) => model.hash === ref.hash) : [];
  const matches =
    byHash.length > 0 || (ref.hash && policy.hashMiss === 'refuse')
      ? byHash
      : ref.name && ref.base
        ? models.filter(
            (model) => model.name === ref.name && model.base === ref.base && (!ref.type || model.type === ref.type)
          )
        : [];

  return matches.length > 1 && policy.ambiguous === 'refuse' ? 'ambiguous' : (matches[0] ?? null);
};

/**
 * Key first, then the content hash when recorded, else name and base (and type when recorded). A model whose files are
 * gone is the last resort: a reinstall leaves the old record beside the new one, and the new one is what the image
 * recorded in effect.
 */
export const resolveRecordedModel = <T extends ResolvableModel>(
  ref: RecordedModelRef,
  models: readonly T[],
  policy: RecordedModelPolicy
): T | 'ambiguous' | null => {
  const present = models.filter((model) => !model.is_missing);
  const byKey = ref.key ? present.find((model) => model.key === ref.key) : undefined;

  if (byKey) {
    return byKey;
  }

  const matched = matchRecordedModel(ref, present, policy);

  if (matched !== null) {
    return matched;
  }

  return (
    (ref.key ? models.find((model) => model.key === ref.key) : undefined) ?? matchRecordedModel(ref, models, policy)
  );
};

export interface ParsedMetadataLoraTarget {
  lowNoiseWeight?: number;
  target?: WanLoraTarget;
}

/** Read optional Wan expert target and differing low-noise weight from a metadata `loras` entry. */
export const readMetadataLoraTarget = (entry: Record<string, unknown>, weight: number): ParsedMetadataLoraTarget => {
  const target = isWanLoraTarget(entry.target) ? entry.target : undefined;
  const lowNoiseWeight =
    target === 'both' &&
    typeof entry.low_noise_weight === 'number' &&
    Number.isFinite(entry.low_noise_weight) &&
    entry.low_noise_weight !== weight
      ? entry.low_noise_weight
      : undefined;

  return {
    ...(target !== undefined ? { target } : {}),
    ...(lowNoiseWeight !== undefined ? { lowNoiseWeight } : {}),
  };
};

/**
 * Fold opposing `'high'` and `'low'` entries for the same LoRA into one `'both'` entry, preserving `weight` for the
 * high-noise expert and adding `lowNoiseWeight` when the low-noise expert differed.
 */
export const mergeOpposingWanLoraTargets = <
  T extends { lowNoiseWeight?: number; target?: WanLoraTarget; weight: number },
>(
  existing: T,
  incoming: { target?: WanLoraTarget; weight: number }
): T | null => {
  if (
    !(
      (existing.target === 'high' && incoming.target === 'low') ||
      (existing.target === 'low' && incoming.target === 'high')
    )
  ) {
    return null;
  }

  const highWeight = existing.target === 'high' ? existing.weight : incoming.weight;
  const lowWeight = existing.target === 'low' ? existing.weight : incoming.weight;
  const { lowNoiseWeight: _previousLow, ...rest } = existing;

  return {
    ...rest,
    target: 'both',
    weight: highWeight,
    ...(lowWeight !== highWeight ? { lowNoiseWeight: lowWeight } : {}),
  } as T;
};

/**
 * How a generation record names a model, and how that name finds an installed model again. Image recall, video recall
 * and Preview Details read the same identifier shapes; only the resolution policy differs between them.
 */

import type { WanLoraTarget } from '@features/generation/contracts';

import { isWanLoraTarget } from '@features/generation/settings';

/**
 * A recorded `ModelIdentifierField` — `{ key, hash, name, base, type }` — with every field optional on read. Records
 * from before identifier fields named the model as `{ model_name, base_model, model_type }`.
 */
export interface RecordedModelRef {
  base: string | null;
  hash: string | null;
  key: string | null;
  name: string | null;
  type: string | null;
}

/** The catalog fields resolution compares; every installed model config carries them. */
interface ResolvableModel {
  base: string;
  hash?: string;
  key: string;
  name: string;
  type: string;
}

/**
 * What resolution does when the exact weights cannot be found.
 *
 * - `hashMiss`: `'refuse'` keeps a recorded hash that matches nothing from falling back to a same-named file, so only
 *   the weights that ran are restored; `'matchName'` lets it fall through to name and base, for installs whose hash
 *   changed on reinstall.
 * - `ambiguous`: `'refuse'` reports a portable identifier several installed models match as `'ambiguous'`;
 *   `'first'` takes the first match in catalog order.
 *
 * Image recall refuses both for the main model, VAE, components, refiner and concepts, and uses
 * `hashMiss: 'matchName'` for reference-image adapters, as legacy web does. Video recall uses
 * `{ hashMiss: 'matchName', ambiguous: 'first' }`.
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

/**
 * Every identifier field the record holds, under its current or legacy name, or null when it is not a record at all.
 * Display reads this; recall needs `toRecordedModelRef`.
 */
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

/**
 * The recorded identifier, or null unless resolution could act on it: an install-local key, a content hash, or a name
 * with its base.
 */
export const toRecordedModelRef = (value: unknown): RecordedModelRef | null => {
  const ref = readRecordedModelRef(value);

  return ref && (ref.key || ref.hash || (ref.name && ref.base)) ? ref : null;
};

/**
 * Resolve the install-local key first, then a portable identifier: the content hash when one was recorded, otherwise
 * name and base, and type when the ref records one. `'ambiguous'` is returned only under `ambiguous: 'refuse'`.
 */
export const resolveRecordedModel = <T extends ResolvableModel>(
  ref: RecordedModelRef,
  models: readonly T[],
  policy: RecordedModelPolicy
): T | 'ambiguous' | null => {
  const byKey = ref.key ? models.find((model) => model.key === ref.key) : undefined;

  if (byKey) {
    return byKey;
  }

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

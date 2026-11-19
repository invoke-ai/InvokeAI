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

/** Key first, then the content hash when recorded, else name and base (and type when recorded). */
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

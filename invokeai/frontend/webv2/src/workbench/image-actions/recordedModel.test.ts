import { describe, expect, it } from 'vitest';

import { resolveRecordedModel, toRecordedModelRef, type RecordedModelPolicy } from './recordedModel';

const model = (key: string, extra: Partial<{ base: string; hash: string; name: string; type: string }> = {}) => ({
  base: 'sdxl',
  key,
  name: 'Juggernaut',
  type: 'main',
  ...extra,
});

const STRICT: RecordedModelPolicy = { ambiguous: 'refuse', hashMiss: 'refuse' };
const LENIENT: RecordedModelPolicy = { ambiguous: 'first', hashMiss: 'matchName' };

describe('toRecordedModelRef', () => {
  it('reads pre-identifier records by their legacy field names', () => {
    expect(toRecordedModelRef({ base_model: 'sdxl', model_name: 'Juggernaut', model_type: 'main' })).toEqual({
      base: 'sdxl',
      hash: null,
      key: null,
      name: 'Juggernaut',
      type: 'main',
    });
  });

  it('accepts only a ref resolution can act on: a key, a hash, or a name with its base', () => {
    expect(toRecordedModelRef({ key: 'k' })).not.toBeNull();
    expect(toRecordedModelRef({ hash: 'blake3:x' })).not.toBeNull();
    expect(toRecordedModelRef({ base: 'sdxl', name: 'Juggernaut' })).not.toBeNull();
    expect(toRecordedModelRef({ name: 'Juggernaut', type: 'main' })).toBeNull();
    expect(toRecordedModelRef('Juggernaut')).toBeNull();
  });
});

describe('resolveRecordedModel', () => {
  const ref = (value: Record<string, unknown>) => toRecordedModelRef(value)!;

  it('prefers the install-local key over every portable identifier', () => {
    const local = model('local', { hash: 'blake3:other', name: 'Renamed' });

    expect(resolveRecordedModel(ref({ hash: 'blake3:x', key: 'local' }), [model('a'), local], STRICT)).toBe(local);
  });

  it('falls back from a missed hash to name and base only under matchName', () => {
    const installed = model('a', { hash: 'blake3:new' });
    const recorded = ref({ base: 'sdxl', hash: 'blake3:old', name: 'Juggernaut' });

    expect(resolveRecordedModel(recorded, [installed], STRICT)).toBeNull();
    expect(resolveRecordedModel(recorded, [installed], LENIENT)).toBe(installed);
  });

  it('refuses or takes the first of several portable matches, as the policy says', () => {
    const first = model('a');
    const second = model('b');
    const recorded = ref({ base: 'sdxl', name: 'Juggernaut' });

    expect(resolveRecordedModel(recorded, [first, second], STRICT)).toBe('ambiguous');
    expect(resolveRecordedModel(recorded, [first, second], LENIENT)).toBe(first);
  });

  it('matches the type only when the ref records one', () => {
    const lora = model('lora', { type: 'lora' });

    expect(resolveRecordedModel(ref({ base: 'sdxl', name: 'Juggernaut' }), [lora], STRICT)).toBe(lora);
    expect(resolveRecordedModel(ref({ base: 'sdxl', name: 'Juggernaut', type: 'main' }), [lora], STRICT)).toBeNull();
  });
});

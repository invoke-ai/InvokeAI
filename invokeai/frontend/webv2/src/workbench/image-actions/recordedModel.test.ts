import { describe, expect, it } from 'vitest';

import { resolveRecordedModel, toRecordedModelRef, type RecordedModelPolicy } from './recordedModel';

const model = (
  key: string,
  extra: Partial<{ base: string; hash: string; is_missing: boolean; name: string; type: string }> = {}
) => ({
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

  it('prefers a present record over a missing one at every step, and takes a missing one only as the last resort', () => {
    const old = model('old', { hash: 'blake3:x', is_missing: true });
    const reinstalled = model('new', { hash: 'blake3:x' });
    const ref = { base: 'sdxl', hash: 'blake3:x', key: 'old', name: 'Juggernaut', type: 'main' };

    // The key names the missing record, the reinstall has the same hash: the present one wins, in either order.
    expect(resolveRecordedModel(ref, [old, reinstalled], STRICT)).toBe(reinstalled);
    expect(resolveRecordedModel(ref, [reinstalled, old], STRICT)).toBe(reinstalled);
    // Several missing matches beside one present one are no ambiguity.
    expect(
      resolveRecordedModel(
        { ...ref, key: null },
        [old, model('old2', { hash: 'blake3:x', is_missing: true }), reinstalled],
        STRICT
      )
    ).toBe(reinstalled);
    // Nothing present matches: the missing record stays what the image names.
    expect(resolveRecordedModel(ref, [old], STRICT)).toBe(old);
    expect(resolveRecordedModel({ ...ref, key: null }, [old], STRICT)).toBe(old);
    // By name and base, too.
    const named = { base: 'sdxl', hash: null, key: null, name: 'Juggernaut', type: null };
    expect(resolveRecordedModel(named, [model('gone', { is_missing: true }), model('here')], LENIENT)).toMatchObject({
      key: 'here',
    });
  });

  it('takes the reinstall of the weights a missing key stood for over another model with the recorded name', () => {
    // Metadata records the key and the name but no hash.
    const ref = { base: 'sdxl', hash: null, key: 'old', name: 'Juggernaut', type: 'main' };
    const missing = model('old', { hash: 'blake3:weights', is_missing: true });
    const reinstall = model('new', { hash: 'blake3:weights', name: 'Juggernaut (renamed)' });
    const namesake = model('namesake', { hash: 'blake3:other' });

    expect(resolveRecordedModel(ref, [missing, namesake, reinstall], STRICT)).toBe(reinstall);
    expect(resolveRecordedModel(ref, [reinstall, namesake, missing], LENIENT)).toBe(reinstall);
    // No reinstall: the recorded identifiers decide, as for any image without a hash.
    expect(resolveRecordedModel(ref, [missing, namesake], STRICT)).toBe(namesake);
    // Two present copies of those weights are as ambiguous as two with the recorded hash.
    expect(resolveRecordedModel(ref, [missing, reinstall, model('copy', { hash: 'blake3:weights' })], STRICT)).toBe(
      'ambiguous'
    );
    expect(resolveRecordedModel(ref, [missing, reinstall, model('copy', { hash: 'blake3:weights' })], LENIENT)).toBe(
      reinstall
    );
  });

  it('counts only the copy of those weights registered for the recorded family and type', () => {
    // One VAE file registered as SDXL and as SD1; the image recorded the SDXL key, now missing, and no hash.
    const vae = (key: string, base: string, extra: Partial<{ is_missing: boolean }> = {}) =>
      model(key, { base, hash: 'blake3:vae', name: 'VAE', type: 'vae', ...extra });
    const missing = vae('old', 'sdxl', { is_missing: true });
    const sdxl = vae('sdxl-copy', 'sdxl');
    const sd1 = vae('sd1-copy', 'sd-1');

    expect(
      resolveRecordedModel(
        { base: 'sdxl', hash: null, key: 'old', name: 'VAE', type: 'vae' },
        [missing, sd1, sdxl],
        STRICT
      )
    ).toBe(sdxl);
    // Without a recorded family, the one the missing key stood for decides.
    expect(
      resolveRecordedModel({ base: null, hash: null, key: 'old', name: null, type: null }, [missing, sd1, sdxl], STRICT)
    ).toBe(sdxl);
  });

  it('matches the type only when the ref records one', () => {
    const lora = model('lora', { type: 'lora' });

    expect(resolveRecordedModel(ref({ base: 'sdxl', name: 'Juggernaut' }), [lora], STRICT)).toBe(lora);
    expect(resolveRecordedModel(ref({ base: 'sdxl', name: 'Juggernaut', type: 'main' }), [lora], STRICT)).toBeNull();
  });
});

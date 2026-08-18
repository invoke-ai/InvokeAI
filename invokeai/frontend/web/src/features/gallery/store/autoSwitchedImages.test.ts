import { describe, expect, it } from 'vitest';

import { createAutoSwitchedImageRegistry } from './autoSwitchedImages';

describe('createAutoSwitchedImageRegistry', () => {
  it('consumes a recorded name exactly once', () => {
    const registry = createAutoSwitchedImageRegistry();
    registry.record('a.png');
    expect(registry.consume('a.png')).toBe(true);
    expect(registry.consume('a.png')).toBe(false);
  });

  it('returns false for a name that was never recorded', () => {
    const registry = createAutoSwitchedImageRegistry();
    expect(registry.consume('a.png')).toBe(false);
  });

  it('consumes entries that render in record order', () => {
    const registry = createAutoSwitchedImageRegistry();
    registry.record('a.png');
    registry.record('b.png');
    expect(registry.consume('a.png')).toBe(true);
    expect(registry.consume('b.png')).toBe(true);
    expect(registry.consume('b.png')).toBe(false);
  });

  it('drops entries recorded before the consumed one — their selections were superseded', () => {
    // Two completions within one thumbnail-fetch window: only b renders. If a's entry survived,
    // it would suppress a genuine user click on a during the next generation.
    const registry = createAutoSwitchedImageRegistry();
    registry.record('a.png');
    registry.record('b.png');
    expect(registry.consume('b.png')).toBe(true);
    expect(registry.consume('a.png')).toBe(false);
  });

  it('clears all pending entries when an unrecorded image renders', () => {
    // A rendered image that was never recorded is user activity: any pending selections have been
    // superseded and will never first-render, so their entries must not linger to swallow a later
    // genuine click.
    const registry = createAutoSwitchedImageRegistry();
    registry.record('a.png');
    expect(registry.consume('user-click.png')).toBe(false);
    registry.record('b.png');
    expect(registry.consume('a.png')).toBe(false);
    // The miss above also settled b.png away.
    expect(registry.consume('b.png')).toBe(false);
  });

  it('lets a genuine re-selection reveal after a stale entry is settled by a later render', () => {
    // JPPhoto's review sequence: a is recorded again after it already rendered (stale entry), then
    // the user selects b -> a during the next generation. The b render must settle the stale entry
    // so the click on a reveals.
    const registry = createAutoSwitchedImageRegistry();
    registry.record('a.png');
    expect(registry.consume('a.png')).toBe(true); // auto-switch renders a
    registry.record('a.png'); // stale re-record
    expect(registry.consume('b.png')).toBe(false); // user clicks b — settles the registry
    expect(registry.consume('a.png')).toBe(false); // user clicks a — reveal must fire
  });

  it('evicts the oldest entry beyond the bound', () => {
    const registry = createAutoSwitchedImageRegistry();
    for (let i = 0; i < 9; i++) {
      registry.record(`image-${i}.png`);
    }
    // 9 recorded, bound is 8 — the oldest is gone.
    expect(registry.consume('image-0.png')).toBe(false);
  });

  it('retains the newest entries under the bound', () => {
    const registry = createAutoSwitchedImageRegistry();
    for (let i = 0; i < 9; i++) {
      registry.record(`image-${i}.png`);
    }
    // image-0 was evicted by the bound, so image-1 is the oldest survivor.
    expect(registry.consume('image-1.png')).toBe(true);
    expect(registry.consume('image-8.png')).toBe(true);
  });

  it('expires entries after the TTL', () => {
    let t = 0;
    const registry = createAutoSwitchedImageRegistry(() => t);
    registry.record('a.png');
    t = 30_001;
    expect(registry.consume('a.png')).toBe(false);
  });

  it('keeps entries up to the TTL boundary', () => {
    let t = 0;
    const registry = createAutoSwitchedImageRegistry(() => t);
    registry.record('a.png');
    t = 30_000;
    expect(registry.consume('a.png')).toBe(true);
  });

  it('prunes expired entries without touching live ones', () => {
    let t = 0;
    const registry = createAutoSwitchedImageRegistry(() => t);
    registry.record('old.png');
    t = 20_000;
    registry.record('new.png');
    t = 40_000;
    // old.png has expired; new.png is still live and consumable.
    expect(registry.consume('new.png')).toBe(true);
  });
});

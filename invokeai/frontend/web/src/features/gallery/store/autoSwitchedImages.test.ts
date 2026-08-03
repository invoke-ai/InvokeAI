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

  it('tracks multiple pending names independently', () => {
    const registry = createAutoSwitchedImageRegistry();
    registry.record('a.png');
    registry.record('b.png');
    expect(registry.consume('b.png')).toBe(true);
    expect(registry.consume('a.png')).toBe(true);
    expect(registry.consume('b.png')).toBe(false);
  });

  it('evicts the oldest entry beyond the bound', () => {
    const registry = createAutoSwitchedImageRegistry();
    for (let i = 0; i < 9; i++) {
      registry.record(`image-${i}.png`);
    }
    // 9 recorded, bound is 8 — the oldest is gone, the rest remain.
    expect(registry.consume('image-0.png')).toBe(false);
    for (let i = 1; i < 9; i++) {
      expect(registry.consume(`image-${i}.png`)).toBe(true);
    }
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
    expect(registry.consume('old.png')).toBe(false);
    expect(registry.consume('new.png')).toBe(true);
  });
});

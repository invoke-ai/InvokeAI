import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { isRevealHolding, revealHoldStore, REVEAL_HOLD_DURATION_MS } from './revealHoldStore';

describe('revealHoldStore', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    revealHoldStore.clear();
    vi.useRealTimers();
  });

  it('holds for the default duration and expires on its own', () => {
    expect(isRevealHolding()).toBe(false);

    revealHoldStore.arm();
    expect(isRevealHolding()).toBe(true);

    vi.advanceTimersByTime(REVEAL_HOLD_DURATION_MS - 1);
    expect(isRevealHolding()).toBe(true);

    vi.advanceTimersByTime(1);
    expect(isRevealHolding()).toBe(false);
  });

  it('re-arming extends the hold instead of expiring early', () => {
    revealHoldStore.arm();
    vi.advanceTimersByTime(REVEAL_HOLD_DURATION_MS - 500);
    revealHoldStore.arm();
    vi.advanceTimersByTime(REVEAL_HOLD_DURATION_MS - 1);
    expect(isRevealHolding()).toBe(true);
    vi.advanceTimersByTime(1);
    expect(isRevealHolding()).toBe(false);
  });

  it('clear releases the hold immediately', () => {
    revealHoldStore.arm();
    revealHoldStore.clear();
    expect(isRevealHolding()).toBe(false);
  });
});

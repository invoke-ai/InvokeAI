import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  createDeferredClear,
  getTerminalProgressAction,
  PROGRESS_IMAGE_RESOLVE_TIMEOUT_MS,
} from './progressImageResolution';

type Event = Parameters<typeof getTerminalProgressAction>[0];

const buildEvent = (overrides: Partial<Event> = {}): Event => ({
  item_id: 1,
  status: 'completed',
  origin: null,
  destination: null,
  ...overrides,
});

const OWNED = { autoSwitch: true, globalProgressItemId: 1 };

describe('getTerminalProgressAction', () => {
  it('defers the clear to the image load for a completed item when auto-switching', () => {
    expect(getTerminalProgressAction(buildEvent(), OWNED)).toBe('arm');
  });

  it('defers the clear when no item owns the shared progress atoms yet', () => {
    expect(getTerminalProgressAction(buildEvent(), { autoSwitch: true, globalProgressItemId: null })).toBe('arm');
  });

  it.each(['canceled', 'failed'] as const)('clears immediately for a %s item, as nothing will load', (status) => {
    expect(getTerminalProgressAction(buildEvent({ status }), OWNED)).toBe('clear');
  });

  it('clears immediately when auto-switch is off, since the viewer will not show the final image', () => {
    expect(getTerminalProgressAction(buildEvent(), { ...OWNED, autoSwitch: false })).toBe('clear');
  });

  it('clears immediately for a canvas item bound for the staging area', () => {
    const event = buildEvent({ origin: 'canvas', destination: 'canvas_session_1' });
    expect(getTerminalProgressAction(event, OWNED)).toBe('clear');
  });

  it('still defers for a canvas item that stays in the viewer', () => {
    const event = buildEvent({ origin: 'canvas', destination: 'canvas' });
    expect(getTerminalProgressAction(event, OWNED)).toBe('arm');
  });

  it('ignores an item that does not own the shared progress atoms', () => {
    // Multi-GPU: canceling item 2 must not blank item 1's still-running preview.
    const event = buildEvent({ item_id: 2, status: 'canceled' });
    expect(getTerminalProgressAction(event, { autoSwitch: true, globalProgressItemId: 1 })).toBe('ignore');
  });

  it('ignores a non-owning item even when it completed successfully', () => {
    const event = buildEvent({ item_id: 2 });
    expect(getTerminalProgressAction(event, { autoSwitch: true, globalProgressItemId: 1 })).toBe('ignore');
  });

  it('uses a backstop long enough not to fire on the normal thumbnail-gated path', () => {
    expect(PROGRESS_IMAGE_RESOLVE_TIMEOUT_MS).toBeGreaterThanOrEqual(5_000);
  });
});

describe('createDeferredClear', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('runs the deadline callback when nothing disarms it', () => {
    const onDeadline = vi.fn();
    const deferred = createDeferredClear(1_000);

    deferred.arm(onDeadline);
    expect(deferred.isArmed()).toBe(true);

    vi.advanceTimersByTime(1_000);

    expect(onDeadline).toHaveBeenCalledOnce();
    expect(deferred.isArmed()).toBe(false);
  });

  it('never runs the deadline callback after a disarm', () => {
    // The regression: item N arms, its final image never loads, item N+1 emits progress (which
    // disarms). N's deadline must not fire later and blank N+1's live preview.
    const onDeadline = vi.fn();
    const deferred = createDeferredClear(1_000);

    deferred.arm(onDeadline);
    deferred.disarm();
    expect(deferred.isArmed()).toBe(false);

    vi.advanceTimersByTime(60_000);

    expect(onDeadline).not.toHaveBeenCalled();
  });

  it('supersedes the previous deadline when re-armed rather than stacking', () => {
    const first = vi.fn();
    const second = vi.fn();
    const deferred = createDeferredClear(1_000);

    deferred.arm(first);
    vi.advanceTimersByTime(900);
    deferred.arm(second);

    // The first deadline's original moment passes with nothing pending for it.
    vi.advanceTimersByTime(100);
    expect(first).not.toHaveBeenCalled();
    expect(second).not.toHaveBeenCalled();

    // The re-arm restarted the clock, so the second fires a full interval after it was armed.
    vi.advanceTimersByTime(900);
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledOnce();
  });

  it('fires at most once per arm', () => {
    const onDeadline = vi.fn();
    const deferred = createDeferredClear(1_000);

    deferred.arm(onDeadline);
    vi.advanceTimersByTime(10_000);

    expect(onDeadline).toHaveBeenCalledOnce();
  });

  it('tolerates disarming when nothing is armed', () => {
    const deferred = createDeferredClear(1_000);

    expect(() => {
      deferred.disarm();
      deferred.disarm();
    }).not.toThrow();
    expect(deferred.isArmed()).toBe(false);
  });

  it('reports not-armed once the deadline has fired, so a late load is a no-op', () => {
    // onLoadImage is gated on isArmed(); a load arriving after the backstop already cleared must
    // not clear a preview that a newer generation has since put up.
    const deferred = createDeferredClear(1_000);
    deferred.arm(vi.fn());
    vi.advanceTimersByTime(1_000);

    expect(deferred.isArmed()).toBe(false);
  });
});

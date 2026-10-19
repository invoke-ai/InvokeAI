import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  consumeQueueItemSwapProgressImage,
  getLatestProgressImage,
  getQueueItemBridgeProgressImage,
  getQueueItemSwapProgressImage,
  progressImageStore,
  SWAP_FRAME_TTL_MS,
} from './progressImageStore';

const frame = (label: string) => ({ dataUrl: `data:image/png;base64,${label}`, height: 32, width: 64 });
const target = (queueItemId: string, itemIndex = 1) => ({ itemIndex, queueItemId });

describe('progressImageStore held frames', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    progressImageStore.clear();
  });

  afterEach(() => {
    progressImageStore.clear();
    vi.useRealTimers();
  });

  it('copies the slot frame into both held sets and keeps the live frame', () => {
    progressImageStore.set(frame('last'), target('queue-1'));

    progressImageStore.hold(target('queue-1'));
    progressImageStore.bindSwapImages('queue-1', ['result.png']);

    expect(getQueueItemBridgeProgressImage('queue-1')).toEqual(frame('last'));
    expect(getQueueItemSwapProgressImage('queue-1', 'result.png')).toEqual(frame('last'));
    // The single-slot preview keeps showing the live frame while routing runs.
    progressImageStore.clear(target('queue-1'));
    expect(getQueueItemBridgeProgressImage('queue-1')).toEqual(frame('last'));
  });

  it('holds nothing for a slot that never produced a frame', () => {
    progressImageStore.hold(target('queue-1'));
    progressImageStore.bindSwapImages('queue-1', ['result.png']);

    expect(getQueueItemBridgeProgressImage('queue-1')).toBeNull();
    expect(getQueueItemSwapProgressImage('queue-1', 'result.png')).toBeNull();
  });

  it('bridges only to the last selected output of a multi-output workflow', () => {
    // Only routed output names may use a held frame; another item in the same batch must not inherit it.
    progressImageStore.set(frame('third'), target('queue-1', 3));
    progressImageStore.hold(target('queue-1', 3));

    expect(getQueueItemSwapProgressImage('queue-1', 'image-3.png')).toBeNull();

    progressImageStore.bindSwapImages('queue-1', ['image-3.png', 'image-3-control.png']);

    expect(getQueueItemSwapProgressImage('queue-1', 'image-3.png')).toBeNull();
    expect(getQueueItemSwapProgressImage('queue-1', 'image-3-control.png')).toEqual(frame('third'));
    expect(getQueueItemSwapProgressImage('queue-1', 'image-1.png')).toBeNull();

    // A later hold for the same queue item starts unbound again.
    progressImageStore.set(frame('fourth'), target('queue-1', 4));
    progressImageStore.hold(target('queue-1', 4));

    expect(getQueueItemSwapProgressImage('queue-1', 'image-3.png')).toBeNull();
  });

  it('consumes the swap frame alone once the finished image has decoded', () => {
    progressImageStore.set(frame('last'), target('queue-1'));
    progressImageStore.hold(target('queue-1'));
    progressImageStore.bindSwapImages('queue-1', ['result.png']);

    consumeQueueItemSwapProgressImage('queue-1');

    expect(getQueueItemSwapProgressImage('queue-1', 'result.png')).toBeNull();
    // The bridge to the batch's next slot is still needed.
    expect(getQueueItemBridgeProgressImage('queue-1')).toEqual(frame('last'));
  });

  it('expires the swap frame so browsing back never replays the low-resolution frame', () => {
    progressImageStore.set(frame('last'), target('queue-1'));
    progressImageStore.hold(target('queue-1'));
    progressImageStore.bindSwapImages('queue-1', ['result.png']);

    vi.advanceTimersByTime(SWAP_FRAME_TTL_MS - 1);
    expect(getQueueItemSwapProgressImage('queue-1', 'result.png')).toEqual(frame('last'));

    vi.advanceTimersByTime(1);
    expect(getQueueItemSwapProgressImage('queue-1', 'result.png')).toBeNull();
    expect(getQueueItemBridgeProgressImage('queue-1')).toEqual(frame('last'));
  });

  it('restarts the expiry when a later slot of the same queue item is held', () => {
    progressImageStore.set(frame('first'), target('queue-1', 1));
    progressImageStore.hold(target('queue-1', 1));
    vi.advanceTimersByTime(SWAP_FRAME_TTL_MS - 1);

    progressImageStore.set(frame('second'), target('queue-1', 2));
    progressImageStore.hold(target('queue-1', 2));
    progressImageStore.bindSwapImages('queue-1', ['second.png']);
    vi.advanceTimersByTime(SWAP_FRAME_TTL_MS - 1);

    expect(getQueueItemSwapProgressImage('queue-1', 'second.png')).toEqual(frame('second'));
  });

  it('keeps only the most recent queue items', () => {
    for (let index = 0; index < 9; index += 1) {
      progressImageStore.set(frame(`frame-${index}`), target(`queue-${index}`));
      progressImageStore.hold(target(`queue-${index}`));
      progressImageStore.bindSwapImages(`queue-${index}`, [`image-${index}.png`]);
    }

    expect(getQueueItemBridgeProgressImage('queue-0')).toBeNull();
    expect(getQueueItemSwapProgressImage('queue-0', 'image-0.png')).toBeNull();
    expect(getQueueItemBridgeProgressImage('queue-1')).toEqual(frame('frame-1'));
    expect(getQueueItemSwapProgressImage('queue-8', 'image-8.png')).toEqual(frame('frame-8'));
  });

  it('forgets a queue item on clearHeld and everything on clear', () => {
    progressImageStore.set(frame('one'), target('queue-1'));
    progressImageStore.hold(target('queue-1'));
    progressImageStore.set(frame('two'), target('queue-2'));
    progressImageStore.hold(target('queue-2'));
    progressImageStore.bindSwapImages('queue-2', ['two.png']);

    progressImageStore.clearHeld('queue-1');
    expect(getQueueItemBridgeProgressImage('queue-1')).toBeNull();
    expect(getQueueItemSwapProgressImage('queue-2', 'two.png')).toEqual(frame('two'));

    progressImageStore.clear();
    expect(getQueueItemBridgeProgressImage('queue-2')).toBeNull();
    expect(getQueueItemSwapProgressImage('queue-2', 'two.png')).toBeNull();
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe('progressImageStore latest frame', () => {
  beforeEach(() => {
    progressImageStore.clear();
  });

  it('falls back to the most recently updated live slot when the latest slot is released', () => {
    progressImageStore.set(frame('video-1'), target('video'));
    progressImageStore.set(frame('anima-1'), target('anima'));
    progressImageStore.set(frame('video-2'), target('video'));
    progressImageStore.set(frame('anima-2'), target('anima'));

    progressImageStore.clear(target('anima'));

    expect(getLatestProgressImage()).toEqual({ ...frame('video-2'), target: target('video') });

    progressImageStore.clear(target('video'));

    expect(getLatestProgressImage()).toBeNull();
  });

  it('keeps the latest frame when a slot that is not the latest is released', () => {
    progressImageStore.set(frame('anima-1'), target('anima'));
    progressImageStore.set(frame('video-1'), target('video'));

    progressImageStore.clear(target('anima'));

    expect(getLatestProgressImage()).toEqual({ ...frame('video-1'), target: target('video') });
  });
});

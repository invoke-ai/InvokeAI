import { beforeEach, describe, expect, it } from 'vitest';

import {
  activeProgressTargetStore,
  getActiveProgressTargets,
  getFollowedProgressTargets,
} from './activeProgressTargetStore';

const target = (queueItemId: string, itemIndex: number) => ({ itemIndex, queueItemId });

describe('activeProgressTargetStore', () => {
  beforeEach(() => {
    activeProgressTargetStore.clear();
  });

  it('holds every concurrently-running target', () => {
    // Keep one live slot per concurrent GPU session; a single value loses other sessions.
    activeProgressTargetStore.set(target('queue-1', 1));
    activeProgressTargetStore.set(target('queue-1', 2));

    expect(getActiveProgressTargets()).toEqual([target('queue-1', 1), target('queue-1', 2)]);
  });

  it('keeps array identity when an already-tracked target reports again', () => {
    activeProgressTargetStore.set(target('queue-1', 1));
    const first = getActiveProgressTargets();

    activeProgressTargetStore.set(target('queue-1', 1));

    // Progress frames arrive many times a second; a new array per frame would
    // re-render every consumer.
    expect(getActiveProgressTargets()).toBe(first);
  });

  it('removes only the settled target', () => {
    activeProgressTargetStore.set(target('queue-1', 1));
    activeProgressTargetStore.set(target('queue-1', 2));

    activeProgressTargetStore.clear(target('queue-1', 1));

    expect(getActiveProgressTargets()).toEqual([target('queue-1', 2)]);
  });

  it('ignores clearing a target that is not tracked', () => {
    activeProgressTargetStore.set(target('queue-1', 1));
    const first = getActiveProgressTargets();

    activeProgressTargetStore.clear(target('queue-9', 9));

    expect(getActiveProgressTargets()).toBe(first);
  });

  it('clears everything when called with no target', () => {
    activeProgressTargetStore.set(target('queue-1', 1));
    activeProgressTargetStore.set(target('queue-2', 1));

    activeProgressTargetStore.clear();

    expect(getActiveProgressTargets()).toEqual([]);
  });

  it('preserves start order so the followed target does not change while it runs', () => {
    // Follow the oldest slot consistently rather than whichever session reports most recently.
    activeProgressTargetStore.set(target('queue-1', 1));
    activeProgressTargetStore.set(target('queue-1', 2));
    activeProgressTargetStore.set(target('queue-1', 1));

    expect(getActiveProgressTargets()[0]).toEqual(target('queue-1', 1));
  });

  it('keeps a settling slot followable but out of the running set', () => {
    // Completed, result not routed yet: the single-slot preview keeps following
    // it, while the tile grid must not count it or a single-GPU batch would
    // flash into two tiles at every item boundary.
    activeProgressTargetStore.set(target('queue-1', 1));
    activeProgressTargetStore.set(target('queue-1', 2));

    activeProgressTargetStore.settle(target('queue-1', 1));

    expect(getActiveProgressTargets()).toEqual([target('queue-1', 2)]);
    // Running first: the settling slot is followed only when nothing is running.
    expect(getFollowedProgressTargets()).toEqual([target('queue-1', 2), target('queue-1', 1)]);

    activeProgressTargetStore.clear(target('queue-1', 1));

    expect(getFollowedProgressTargets()).toEqual([target('queue-1', 2)]);
  });

  it('ignores settling a slot that never reported progress', () => {
    activeProgressTargetStore.set(target('queue-1', 1));
    const first = getFollowedProgressTargets();

    activeProgressTargetStore.settle(target('queue-1', 2));

    expect(getFollowedProgressTargets()).toBe(first);
  });

  it('returns a settled slot to the running set when it reports progress again', () => {
    activeProgressTargetStore.set(target('queue-1', 1));
    activeProgressTargetStore.settle(target('queue-1', 1));

    activeProgressTargetStore.set(target('queue-1', 1));

    expect(getActiveProgressTargets()).toEqual([target('queue-1', 1)]);
    expect(getFollowedProgressTargets()).toEqual([target('queue-1', 1)]);
  });

  it('clears settling slots along with running ones', () => {
    activeProgressTargetStore.set(target('queue-1', 1));
    activeProgressTargetStore.settle(target('queue-1', 1));

    activeProgressTargetStore.clear();

    expect(getFollowedProgressTargets()).toEqual([]);
  });
});

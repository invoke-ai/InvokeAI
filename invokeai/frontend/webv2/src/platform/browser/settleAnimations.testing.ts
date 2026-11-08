/**
 * Wait for real browser animations before visual assertions; visible elements can still be mid-transition and
 * yield false contrast failures.
 */

/**
 * Skip infinite iterations/duration, paused animations, and zero playback rate; each can leave finished pending
 * forever.
 */
const willFinish = (animation: Animation): boolean => {
  const timing = animation.effect?.getComputedTiming();
  const duration = typeof timing?.duration === 'number' ? timing.duration : 0;

  return (
    timing?.iterations !== Infinity &&
    Number.isFinite(duration) &&
    animation.playbackRate !== 0 &&
    animation.playState !== 'paused'
  );
};

/**
 * Settle the whole document by default, including sibling backdrops and portals. Ignore nonterminating animations;
 * use allSettled because removal cancels CSS animations and rejects finished.
 */
export const settleAnimations = async (root: Document | Element = document): Promise<void> => {
  await Promise.allSettled(
    root
      .getAnimations({ subtree: true })
      .filter(willFinish)
      .map((animation) => animation.finished)
  );
};

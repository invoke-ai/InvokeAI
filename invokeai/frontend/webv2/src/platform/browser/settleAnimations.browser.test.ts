import { afterEach, beforeEach, expect, it } from 'vitest';

import { settleAnimations } from './settleAnimations.testing';

/**
 * Use the test runner's deadline for never-ending cases; extra timing thresholds introduce load-sensitive
 * failures.
 */

let host: HTMLDivElement | null = null;
let styles: HTMLStyleElement | null = null;
/** A second container, for animations that must sit OUTSIDE the subtree a test passes in. */
let detached: HTMLDivElement;

const FADE: Keyframe[] = [{ opacity: 0 }, { opacity: 1 }];
const CSS_FADE_MS = 400;

const container = (): HTMLDivElement => (host ??= document.body.appendChild(document.createElement('div')));

const scripted = (options: KeyframeAnimationOptions): Animation =>
  container().appendChild(document.createElement('div')).animate(FADE, options);

/** Use CSS animations so element removal rejects finished, matching real components. */
const cssAnimated = (): { animation: Animation; element: HTMLDivElement } => {
  styles ??= document.head.appendChild(document.createElement('style'));
  styles.textContent = `@keyframes settle-probe { from { opacity: 0 } to { opacity: 1 } }
    .settle-probe { animation: settle-probe ${String(CSS_FADE_MS)}ms }`;

  const element = container().appendChild(document.createElement('div'));

  element.className = 'settle-probe';

  return { animation: element.getAnimations()[0]!, element };
};

beforeEach(() => {
  detached = document.body.appendChild(document.createElement('div'));
});

afterEach(() => {
  detached.remove();
  host?.remove();
  styles?.remove();
  host = null;
  styles = null;
});

it('waits for a running animation in the subtree rather than returning on the first frame', async () => {
  const animation = scripted({ duration: 120, fill: 'forwards' });

  await settleAnimations(host!);

  expect(animation.playState).toBe('finished');
});

it('finds animations anywhere in the document when given no subtree, since backdrops and menus are portalled', async () => {
  // Cover sibling backdrops and document-root animations, not only the audited subtree or body.
  const outside = detached.appendChild(document.createElement('div'));
  const sibling = outside.animate(FADE, { duration: 120, fill: 'forwards' });
  // Keep the root animation longest so a body-only scan would return early.
  const root = document.documentElement.animate([{ opacity: 0.99 }, { opacity: 1 }], { duration: 400 });

  await settleAnimations();

  expect([sibling.playState, root.playState]).toEqual(['finished', 'finished']);
});

it('skips an animation that repeats forever instead of hanging on one that never finishes', async () => {
  scripted({ duration: 50, iterations: Infinity });
  const finite = scripted({ duration: 50, fill: 'forwards' });

  await settleAnimations(host!);

  expect(finite.playState).toBe('finished');
});

it.each([
  ['paused', (animation: Animation) => animation.pause()],
  ['stopped by a zero playback rate', (animation: Animation) => (animation.playbackRate = 0)],
])('skips an animation %s, which reports finite timing but never settles', async (_case, disrupt) => {
  // These report `iterations: 1`, so the obvious repeats-forever check does not catch them.
  disrupt(scripted({ duration: 50 }));
  const finite = scripted({ duration: 50, fill: 'forwards' });

  await settleAnimations(host!);

  expect(finite.playState).toBe('finished');
});

it('skips an animation with an infinite duration, which also reports finite iterations', async () => {
  scripted({ duration: Infinity });
  const finite = scripted({ duration: 50, fill: 'forwards' });

  await settleAnimations(host!);

  expect(finite.playState).toBe('finished');
});

it('settles when an animating element is removed mid-flight, which rejects rather than finishes', async () => {
  const { animation, element } = cssAnimated();
  const rejection = expect(animation.finished).rejects.toThrow();
  const settled = settleAnimations(host!);

  element.remove();

  // Resolve before the fade duration to prove cancellation, not natural completion, settled the wait.
  await expect(settled).resolves.toBeUndefined();
  await rejection;
});

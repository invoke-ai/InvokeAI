import { expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

/*
 * Guards scripts/browser-test-viewport.ts. The two tests run in order: the first leaves the runner's outer page
 * scrolled, as any scrollIntoView or focus() in a test can, and the second needs the reset before it to reach an
 * element at the iframe's top edge. Without the reset the hover times out, as FeatureHint's suite did on CI.
 */

it('scrolls the outer page from inside the test iframe', () => {
  const below = document.createElement('div');

  below.style.cssText = 'position:absolute;left:0;top:1200px;width:10px;height:10px';
  document.body.append(below);
  below.scrollIntoView();
  below.remove();
  window.scrollTo(0, 0);

  expect(window.parent.scrollY).toBeGreaterThan(0);
});

it('starts the next test with an element at the iframe corner reachable', async () => {
  const corner = document.createElement('div');

  corner.style.cssText = 'position:fixed;left:0;top:0;width:2px;height:2px;z-index:2147483647';
  document.body.append(corner);

  try {
    await userEvent.hover(corner, { timeout: 2000 });
  } finally {
    corner.remove();
  }
});

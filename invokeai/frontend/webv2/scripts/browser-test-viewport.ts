import { beforeEach } from 'vitest';

/**
 * Start every browser test with the pages around the test iframe scrolled to the origin.
 *
 * The runner's outer page is a few pixels taller than its viewport, so a `scrollIntoView` or `focus()` inside a
 * test scrolls it too, and that offset outlives the test file. Playwright judges hover and click targets against the
 * top-level viewport, so a later file's elements at the iframe's top edge then sit outside it and every pointer
 * action on them times out (FeatureHint's corner parking element fails all five of its tests this way). The test's
 * own document is left alone: its scroll belongs to the test.
 */
beforeEach(() => {
  let frame: Window = window;

  while (frame.parent !== frame) {
    frame = frame.parent;

    try {
      frame.scrollTo(0, 0);
    } catch {
      // A cross-origin ancestor cannot be scrolled from here, and nothing inside it can leak into this one.
      return;
    }
  }
});

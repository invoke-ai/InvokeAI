import { expect, it } from 'vitest';

/*
 * Playwright judges hover and click targets against the runner's outer page. If that page can scroll, a
 * scrollIntoView or focus() in one test leaves it offset for later files, and their pointer actions on elements at
 * the iframe's top edge time out (FeatureHint's suite fails this way). The runner keeps that page unscrollable by
 * laying the test iframe out as a block exactly filling it; an inline iframe leaves a few pixels of overflow below.
 */

it('keeps the outer page unscrolled when the test iframe scrolls', () => {
  const below = document.createElement('input');

  below.style.cssText = 'position:absolute;left:0;top:5000px';
  document.body.append(below);

  try {
    below.scrollIntoView();
    below.focus();
    expect(window.scrollY).toBeGreaterThan(0);
    expect(window.parent.scrollY).toBe(0);
  } finally {
    below.remove();
    window.scrollTo(0, 0);
  }
});

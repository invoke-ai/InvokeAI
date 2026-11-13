import { expect, it } from 'vitest';

/*
 * Playwright judges hover and click targets against the runner's outer page. If that page can scroll, a
 * scrollIntoView or focus() in one test leaves it offset for later files, and their pointer actions on elements at
 * the iframe's top edge time out. A viewport change can temporarily make the iframe taller than its parent;
 * the runner must clip that overflow without preventing scrolling inside the test document.
 */

it.each([0, 176])('keeps the outer page unscrolled with %ipx of iframe overflow', (overflow) => {
  const below = document.createElement('input');
  const frame = window.frameElement as HTMLIFrameElement;
  const previousHeight = frame.style.height;

  expect(window.innerWidth).toBe(window.parent.innerWidth);
  expect(window.innerHeight).toBe(window.parent.innerHeight);
  if (overflow > 0) {
    frame.style.height = `${window.parent.innerHeight + overflow}px`;
  }
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
    frame.style.height = previousHeight;
  }
});

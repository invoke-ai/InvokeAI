import { describe, expect, it } from 'vitest';

import { isModalPresent, registerModalPresence } from './modalPresence';

describe('modal presence', () => {
  it('stays present until every registration is released, whatever the order', () => {
    const releaseOuter = registerModalPresence();
    const releaseInner = registerModalPresence();

    releaseOuter();
    expect(isModalPresent()).toBe(true);

    releaseInner();
    expect(isModalPresent()).toBe(false);
  });

  it('ignores a repeated release instead of releasing another modal', () => {
    const releaseOuter = registerModalPresence();
    const releaseInner = registerModalPresence();

    releaseInner();
    releaseInner();
    expect(isModalPresent()).toBe(true);

    releaseOuter();
    expect(isModalPresent()).toBe(false);
  });

  it('settles on one registration across a StrictMode mount, cleanup and remount', () => {
    const releaseFirstMount = registerModalPresence();
    releaseFirstMount();
    const releaseRemount = registerModalPresence();

    // A stale cleanup from the discarded mount must not release the live one.
    releaseFirstMount();
    expect(isModalPresent()).toBe(true);

    // Unmounting without closing releases through the same cleanup.
    releaseRemount();
    expect(isModalPresent()).toBe(false);
  });
});

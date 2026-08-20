import { afterEach, describe, expect, it, vi } from 'vitest';

import { abortAndWaitForPendingRefreshes, openMediaInNewTab } from './useMediaCookieRefresh';

describe('openMediaInNewTab', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('opens the media URL directly when cookie self-heal is not pending', () => {
    const open = vi.fn(() => null);
    vi.stubGlobal('window', { open });

    openMediaInNewTab('api/v1/images/i/test.png/full');

    expect(open).toHaveBeenCalledWith('api/v1/images/i/test.png/full', '_blank', 'noopener,noreferrer');
  });
});

describe('media cookie refresh shutdown', () => {
  it('aborts a stalled request instead of blocking logout indefinitely', async () => {
    let settle: (() => void) | undefined;
    const promise = new Promise<void>((resolve) => {
      settle = resolve;
    });
    const abort = vi.fn(() => settle?.());
    const pending = new Set([{ promise, abort }]);

    await abortAndWaitForPendingRefreshes(pending);

    expect(abort).toHaveBeenCalledOnce();
  });

  it('is a no-op when no refresh is pending', async () => {
    await expect(abortAndWaitForPendingRefreshes(new Set())).resolves.toBeUndefined();
  });
});

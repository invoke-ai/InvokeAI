import { describe, expect, it, vi } from 'vitest';

import { abortAndWaitForPendingRefreshes } from './useMediaCookieRefresh';

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

import { describe, expect, it, vi } from 'vitest';

import { logoutAfterServerConfirmation } from './logoutAfterServerConfirmation';

describe('logoutAfterServerConfirmation', () => {
  it('clears local authentication after the server confirms logout', async () => {
    const clearLocalAuthentication = vi.fn();

    await logoutAfterServerConfirmation(() => Promise.resolve(), clearLocalAuthentication);

    expect(clearLocalAuthentication).toHaveBeenCalledOnce();
  });

  it('retains local authentication when server logout fails', async () => {
    const clearLocalAuthentication = vi.fn();

    await expect(
      logoutAfterServerConfirmation(() => Promise.reject(new Error('network error')), clearLocalAuthentication)
    ).rejects.toThrow('network error');
    expect(clearLocalAuthentication).not.toHaveBeenCalled();
  });

  it('waits for media-cookie refreshes before logging out', async () => {
    const resumeMediaCookieRefresh = vi.fn();
    let settleRefresh: (() => void) | undefined;
    const pauseMediaCookieRefresh = vi.fn(
      () =>
        new Promise<() => void>((resolve) => {
          settleRefresh = () => resolve(resumeMediaCookieRefresh);
        })
    );
    const logoutOnServer = vi.fn(() => Promise.resolve());
    const clearLocalAuthentication = vi.fn();

    const logoutPromise = logoutAfterServerConfirmation(
      logoutOnServer,
      clearLocalAuthentication,
      pauseMediaCookieRefresh
    );

    expect(logoutOnServer).not.toHaveBeenCalled();
    settleRefresh?.();
    await logoutPromise;

    expect(logoutOnServer).toHaveBeenCalledOnce();
    expect(clearLocalAuthentication).toHaveBeenCalledOnce();
    expect(resumeMediaCookieRefresh).not.toHaveBeenCalled();
  });

  it('resumes media-cookie refreshes when server logout fails', async () => {
    const resumeMediaCookieRefresh = vi.fn();
    const pauseMediaCookieRefresh = vi.fn(() => Promise.resolve(resumeMediaCookieRefresh));

    await expect(
      logoutAfterServerConfirmation(() => Promise.reject(new Error('network error')), vi.fn(), pauseMediaCookieRefresh)
    ).rejects.toThrow('network error');

    expect(resumeMediaCookieRefresh).toHaveBeenCalledOnce();
  });
});

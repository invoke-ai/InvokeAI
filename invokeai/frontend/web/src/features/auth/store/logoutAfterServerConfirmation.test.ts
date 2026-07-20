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
});

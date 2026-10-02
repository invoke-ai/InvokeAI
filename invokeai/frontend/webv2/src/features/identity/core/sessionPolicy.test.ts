import { describe, expect, it } from 'vitest';

import { shouldExpireUnauthorizedSession } from './sessionPolicy';

describe('unauthorized session policy', () => {
  it('expires an authenticated multi-user session but not login or single-user requests', () => {
    expect(
      shouldExpireUnauthorizedSession({ multiuserEnabled: true, phase: 'ready', user: { user_id: 'admin' } })
    ).toBe(true);
    expect(shouldExpireUnauthorizedSession({ multiuserEnabled: true, phase: 'ready', user: null })).toBe(false);
    expect(shouldExpireUnauthorizedSession({ multiuserEnabled: false, phase: 'ready', user: null })).toBe(false);
  });

  it('rejects the current stored credential while auth mode is unresolved', () => {
    expect(shouldExpireUnauthorizedSession({ multiuserEnabled: false, phase: 'unknown', user: null })).toBe(true);
    expect(shouldExpireUnauthorizedSession({ multiuserEnabled: false, phase: 'unavailable', user: null })).toBe(true);
  });
});

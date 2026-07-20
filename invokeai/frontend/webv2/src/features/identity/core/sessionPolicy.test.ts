import { describe, expect, it } from 'vitest';

import { shouldExpireUnauthorizedSession } from './sessionPolicy';

describe('unauthorized session policy', () => {
  it('expires only an authenticated multi-user session', () => {
    expect(shouldExpireUnauthorizedSession({ multiuserEnabled: true, user: { user_id: 'admin' } })).toBe(true);
    expect(shouldExpireUnauthorizedSession({ multiuserEnabled: true, user: null })).toBe(false);
    expect(shouldExpireUnauthorizedSession({ multiuserEnabled: false, user: null })).toBe(false);
  });
});

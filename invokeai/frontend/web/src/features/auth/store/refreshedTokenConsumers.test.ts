import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { authSliceConfig, currentUserUpdated, setCredentials, tokenRefreshed } from './authSlice';

const user = {
  user_id: 'user',
  email: 'user@example.com',
  display_name: null,
  is_admin: false,
  is_active: true,
};

describe('refreshed token consumers', () => {
  it('updates the Redux token used by socket reconnects', () => {
    let state = authSliceConfig.slice.reducer(undefined, setCredentials({ token: 'old', user }));
    state = authSliceConfig.slice.reducer(state, tokenRefreshed('new'));

    expect(state.token).toBe('new');
  });

  it('updates profile data without replacing the refreshed token', () => {
    let state = authSliceConfig.slice.reducer(undefined, setCredentials({ token: 'new', user }));
    state = authSliceConfig.slice.reducer(state, currentUserUpdated({ ...user, display_name: 'Updated' }));

    expect(state.token).toBe('new');
    expect(state.user?.display_name).toBe('Updated');

    const profileSource = readFileSync(
      fileURLToPath(new URL('../components/UserProfile.tsx', import.meta.url)),
      'utf8'
    );
    expect(profileSource).toContain('currentUserUpdated({');
    expect(profileSource).not.toContain('setCredentials({');
  });

  it('commits a refreshed bearer token only after its media cookie is synchronized', () => {
    const apiSource = readFileSync(fileURLToPath(new URL('../../../services/api/index.ts', import.meta.url)), 'utf8');
    const cookieWrite = apiSource.indexOf('const mediaCookieResponse = await fetch');
    const tokenCommit = apiSource.indexOf('api.dispatch(tokenRefreshed(refreshedToken))');

    expect(cookieWrite).toBeGreaterThan(-1);
    expect(tokenCommit).toBeGreaterThan(cookieWrite);
    expect(apiSource).toContain('if (!mediaCookieResponse.ok');
    expect(apiSource.slice(cookieWrite, tokenCommit)).toContain('/api/v1/auth/media-cookie');
  });
});

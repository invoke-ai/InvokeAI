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

  it('attempts the media-cookie sync before committing a refreshed bearer token', () => {
    const apiSource = readFileSync(fileURLToPath(new URL('../../../services/api/index.ts', import.meta.url)), 'utf8');
    const cookieWrite = apiSource.indexOf('const mediaCookieResponse = await fetch');
    const tokenCommit = apiSource.indexOf('api.dispatch(tokenRefreshed(refreshedToken))');

    expect(cookieWrite).toBeGreaterThan(-1);
    expect(tokenCommit).toBeGreaterThan(cookieWrite);
    expect(apiSource.slice(cookieWrite, tokenCommit)).toContain('/api/v1/auth/media-cookie');
    // The commit is refused only when the server rejects the refreshed token itself
    // (401/403); a 5xx or network failure must NOT discard the token — dropping it
    // would hard-expire an active session over a transient cookie-endpoint problem,
    // while the media cookie itself self-heals on later refreshes.
    expect(apiSource.slice(cookieWrite, tokenCommit)).toContain('mediaCookieResponse.status === 401');
    // The sync fetch is time-bounded so the exclusive cross-tab media-auth lock
    // (shared with login/logout) can never be held indefinitely by a stalled request.
    expect(apiSource).toContain('AbortSignal.timeout(MEDIA_COOKIE_SYNC_TIMEOUT_MS)');
  });
});

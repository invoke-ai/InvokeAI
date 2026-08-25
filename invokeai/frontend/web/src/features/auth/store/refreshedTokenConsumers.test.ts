import { Buffer } from 'node:buffer';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import {
  authSliceConfig,
  currentUserUpdated,
  externalTokenAdopted,
  getTokenSessionKey,
  setCredentials,
  tokenRefreshed,
} from './authSlice';

const user = {
  user_id: 'user',
  email: 'user@example.com',
  display_name: null,
  is_admin: false,
  is_active: true,
};

const tokenFor = (userId: string, nonce = 0, epoch = 0) =>
  `header.${Buffer.from(JSON.stringify({ user_id: userId, nonce, token_epoch: epoch })).toString('base64url')}.signature`;

describe('refreshed token consumers', () => {
  it('updates the Redux token the next request will carry', () => {
    let state = authSliceConfig.slice.reducer(undefined, setCredentials({ token: 'old', user }));
    state = authSliceConfig.slice.reducer(state, tokenRefreshed('new'));

    expect(state.token).toBe('new');
  });

  it('keeps one session key across a refresh, so the socket is not rebuilt for it', () => {
    // The middleware mints new bytes for the same login on every mutating request. Consumers that
    // key work to a session — `useSocketIO` — must not see that as a new session.
    expect(getTokenSessionKey(tokenFor('user', 1))).toBe(getTokenSessionKey(tokenFor('user', 2)));
    expect(getTokenSessionKey(tokenFor('user', 1))).not.toBe(getTokenSessionKey(tokenFor('other', 1)));
    expect(getTokenSessionKey(null)).toBeNull();
  });

  it('changes the session key when a revoked epoch supersedes the token', () => {
    // A password change bumps the epoch and the server drops every socket authenticated under the
    // old one; socket.io does not retry that, so the replacement token must count as a new session.
    expect(getTokenSessionKey(tokenFor('user', 1, 0))).not.toBe(getTokenSessionKey(tokenFor('user', 2, 1)));
  });

  it('falls back to the token itself when it carries no user id', () => {
    // Nothing to compare identities with, so byte equality is the only safe answer.
    expect(getTokenSessionKey('opaque-token')).toBe('opaque-token');
    expect(getTokenSessionKey('')).toBeNull();
  });

  it('clears the previous user when adopting a token from another tab', () => {
    let state = authSliceConfig.slice.reducer(undefined, setCredentials({ token: tokenFor(user.user_id), user }));
    state = authSliceConfig.slice.reducer(state, externalTokenAdopted(tokenFor('other-user')));

    expect(state.token).toBe(tokenFor('other-user'));
    expect(state.user).toBeNull();
    expect(state.isAuthenticated).toBe(true);
  });

  it('preserves the current user when another tab refreshes the same account token', () => {
    let state = authSliceConfig.slice.reducer(undefined, setCredentials({ token: tokenFor(user.user_id), user }));
    state = authSliceConfig.slice.reducer(state, externalTokenAdopted(tokenFor(user.user_id)));

    expect(state.token).toBe(tokenFor(user.user_id));
    expect(state.user).toEqual(user);
    expect(state.isAuthenticated).toBe(true);
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
    const tokenCommit = apiSource.indexOf('dispatch(tokenRefreshed(refreshedToken))');

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

  it('adopts a refreshed token written by another tab', () => {
    const protectedRouteSource = readFileSync(
      fileURLToPath(new URL('../components/ProtectedRoute.tsx', import.meta.url)),
      'utf8'
    );

    expect(protectedRouteSource).toContain("localStorage.getItem('auth_token')");
    expect(protectedRouteSource).toContain('dispatch(externalTokenAdopted(');
  });
});

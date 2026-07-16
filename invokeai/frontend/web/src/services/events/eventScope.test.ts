import { describe, expect, it } from 'vitest';

import { getEventScope, REDACTED_USER_ID } from './eventScope';

const buildGetState = (user: { user_id: string; is_admin: boolean } | null, token: string | null = null) =>
  (() => ({ auth: { user, token } })) as never;

describe('getEventScope', () => {
  it("classifies the current user's event as own", () => {
    const getState = buildGetState({ user_id: 'user-1', is_admin: false }, 'token-1');
    expect(getEventScope(getState, { user_id: 'user-1' })).toBe('own');
  });

  it("classifies another user's event as foreign", () => {
    const getState = buildGetState({ user_id: 'admin-1', is_admin: true }, 'token-1');
    expect(getEventScope(getState, { user_id: 'user-1' })).toBe('foreign');
  });

  it('classifies redacted companion events as sanitized, regardless of the current user', () => {
    expect(getEventScope(buildGetState({ user_id: 'user-1', is_admin: false }), { user_id: REDACTED_USER_ID })).toBe(
      'sanitized'
    );
    expect(getEventScope(buildGetState(null), { user_id: REDACTED_USER_ID })).toBe('sanitized');
  });

  it('classifies every event as own in single-user mode (no token, no authenticated user)', () => {
    const getState = buildGetState(null);
    expect(getEventScope(getState, { user_id: 'anyone' })).toBe('own');
    expect(getEventScope(getState, {})).toBe('own');
  });

  it('classifies events as foreign while multiuser auth is hydrating (token present, user not yet loaded)', () => {
    // The socket connects with the localStorage token before /me populates auth.user. In that
    // window an admin client already receives other users' events via the admin room; they must
    // not be treated as the client's own.
    const getState = buildGetState(null, 'token-1');
    expect(getEventScope(getState, { user_id: 'other-user' })).toBe('foreign');
    expect(getEventScope(getState, { user_id: 'anyone' })).toBe('foreign');
  });
});

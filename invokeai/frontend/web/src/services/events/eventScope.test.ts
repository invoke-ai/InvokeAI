import { describe, expect, it } from 'vitest';

import { getEventScope, REDACTED_USER_ID } from './eventScope';

const buildGetState = (user: { user_id: string; is_admin: boolean } | null) => (() => ({ auth: { user } })) as never;

describe('getEventScope', () => {
  it("classifies the current user's event as own", () => {
    const getState = buildGetState({ user_id: 'user-1', is_admin: false });
    expect(getEventScope(getState, { user_id: 'user-1' })).toBe('own');
  });

  it("classifies another user's event as foreign", () => {
    const getState = buildGetState({ user_id: 'admin-1', is_admin: true });
    expect(getEventScope(getState, { user_id: 'user-1' })).toBe('foreign');
  });

  it('classifies redacted companion events as sanitized, regardless of the current user', () => {
    expect(getEventScope(buildGetState({ user_id: 'user-1', is_admin: false }), { user_id: REDACTED_USER_ID })).toBe(
      'sanitized'
    );
    expect(getEventScope(buildGetState(null), { user_id: REDACTED_USER_ID })).toBe('sanitized');
  });

  it('classifies every event as own in single-user mode (no authenticated user)', () => {
    const getState = buildGetState(null);
    expect(getEventScope(getState, { user_id: 'anyone' })).toBe('own');
    expect(getEventScope(getState, {})).toBe('own');
  });
});

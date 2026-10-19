import { describe, expect, it } from 'vitest';

import { classifyStoredCredential, readTokenUserId } from './storedCredential';

/** JWT segments are base64url-encoded UTF-8 JSON. */
const encodeSegment = (value: unknown): string =>
  btoa(String.fromCharCode(...new TextEncoder().encode(JSON.stringify(value))))
    .replaceAll('=', '')
    .replaceAll('+', '-')
    .replaceAll('/', '_');

const tokenFor = (claims: unknown): string => `${encodeSegment({ alg: 'HS256' })}.${encodeSegment(claims)}.signature`;

describe('stored credential classification', () => {
  it('reads the user id claim from base64url payloads, including non-ASCII claims', () => {
    // '~~~?' encodes to a base64url '-' that plain atob rejects.
    expect(readTokenUserId(tokenFor({ padding: '~~~?', user_id: 'user-a' }))).toBe('user-a');
    expect(readTokenUserId(tokenFor({ email: 'zoë@example.com', user_id: 'zoë' }))).toBe('zoë');
  });

  it('cannot vouch for opaque, malformed or claimless tokens', () => {
    expect(readTokenUserId('opaque-token')).toBeNull();
    expect(readTokenUserId('header.%%%.signature')).toBeNull();
    expect(readTokenUserId(tokenFor({ sub: 'user-a' }))).toBeNull();
    expect(readTokenUserId(tokenFor({ user_id: 7 }))).toBeNull();
  });

  it('adopts silently only a token the claim ties to the active principal', () => {
    const renewed = tokenFor({ user_id: 'user-a' });

    expect(classifyStoredCredential('held', 'held', 'user-a')).toBe('unchanged');
    expect(classifyStoredCredential(null, 'held', 'user-a')).toBe('removed');
    expect(classifyStoredCredential(renewed, 'held', 'user-a')).toBe('renewed');
    expect(classifyStoredCredential(tokenFor({ user_id: 'user-b' }), 'held', 'user-a')).toBe('principal-changed');
    expect(classifyStoredCredential('opaque-token', 'held', 'user-a')).toBe('principal-changed');
    // Without an active principal there is nothing to renew: a signed-out tab follows a sign-in.
    expect(classifyStoredCredential(renewed, null, null)).toBe('principal-changed');
    expect(classifyStoredCredential(renewed, 'held', null)).toBe('principal-changed');
  });
});

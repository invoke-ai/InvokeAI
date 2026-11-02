/**
 * How the credential shared through browser storage differs from the one this tab holds.
 *
 * - `unchanged`: storage holds this tab's token.
 * - `removed`: another tab signed out.
 * - `renewed`: another tab replaced the token for the same principal (sliding renewal or a password change).
 * - `principal-changed`: another account, or a token whose principal cannot be read.
 */
export type StoredCredentialChange = 'unchanged' | 'removed' | 'renewed' | 'principal-changed';

const decodeBase64Url = (value: string): string => {
  const base64 = value.replaceAll('-', '+').replaceAll('_', '/');
  const binary = atob(base64.padEnd(base64.length + ((4 - (base64.length % 4)) % 4), '='));

  return new TextDecoder().decode(Uint8Array.from(binary, (character) => character.charCodeAt(0)));
};

const readTokenClaims = (token: string): object | null => {
  const payload = token.split('.')[1];

  if (payload === undefined) {
    return null;
  }

  try {
    const claims: unknown = JSON.parse(decodeBase64Url(payload));

    return typeof claims === 'object' && claims !== null ? claims : null;
  } catch {
    return null;
  }
};

/** Read the backend's `user_id` claim without verifying it; the backend remains the authority for who a token is. */
export const readTokenUserId = (token: string): string | null => {
  const claims = readTokenClaims(token);

  return claims !== null && 'user_id' in claims && typeof claims.user_id === 'string' ? claims.user_id : null;
};

/**
 * When the token stops being honored, in epoch milliseconds, from its unverified `exp` claim (seconds); null when it
 * carries none. Only a hint for warning the user: the backend decides, and renews it on activity.
 */
export const readTokenExpiry = (token: string): number | null => {
  const claims = readTokenClaims(token);

  return claims !== null && 'exp' in claims && typeof claims.exp === 'number' && Number.isFinite(claims.exp)
    ? claims.exp * 1000
    : null;
};

/**
 * The claim only decides whether a renewal can be adopted without a request. Anything it cannot vouch for is a
 * principal change, which resolves the principal from the backend.
 */
export const classifyStoredCredential = (
  stored: string | null,
  held: string | null,
  principalId: string | null
): StoredCredentialChange => {
  if (stored === held) {
    return 'unchanged';
  }

  if (stored === null) {
    return 'removed';
  }

  return held !== null && principalId !== null && readTokenUserId(stored) === principalId
    ? 'renewed'
    : 'principal-changed';
};

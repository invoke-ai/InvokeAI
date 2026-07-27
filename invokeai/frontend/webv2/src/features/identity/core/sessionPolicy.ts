export interface UnauthorizedSessionState {
  phase: 'unknown' | 'unavailable' | 'ready';
  multiuserEnabled: boolean;
  user: unknown | null;
}

/**
 * The transport only calls this policy for the current stored credential.
 * Rejecting that credential while auth mode is unresolved must clear it;
 * login requests have no stored bearer token and never reach this policy.
 */
export const shouldExpireUnauthorizedSession = (session: UnauthorizedSessionState): boolean =>
  session.phase !== 'ready' || (session.multiuserEnabled && session.user !== null);

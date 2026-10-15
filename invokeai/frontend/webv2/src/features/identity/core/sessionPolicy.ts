export interface UnauthorizedSessionState {
  phase: 'unknown' | 'unavailable' | 'ready';
  multiuserEnabled: boolean;
  user: unknown | null;
}

/**
 * Identity only applies this policy to a 401 for the credential the tab currently holds.
 * Rejecting that credential while auth mode is unresolved must clear it;
 * login requests carry no bearer token and never reach this policy.
 */
export const shouldExpireUnauthorizedSession = (session: UnauthorizedSessionState): boolean =>
  session.phase !== 'ready' || (session.multiuserEnabled && session.user !== null);

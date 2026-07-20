export interface UnauthorizedSessionState {
  multiuserEnabled: boolean;
  user: unknown | null;
}

/** Login failures and single-user requests must never expire the active shell session. */
export const shouldExpireUnauthorizedSession = (session: UnauthorizedSessionState): boolean =>
  session.multiuserEnabled && session.user !== null;

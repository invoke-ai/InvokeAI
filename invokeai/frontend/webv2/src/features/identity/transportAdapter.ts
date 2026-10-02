import { captureAccountScope, isAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';

import type { IdentityTokenAdapter } from './core/tokenStorage';

import { browserIdentityTokenAdapter } from './core/tokenStorage';
import { handleUnauthorizedResponse } from './session';

export interface IdentityTransportAuthAdapter {
  getIdentity(): unknown;
  getToken(): string | null;
  onUnauthorized(rejectedToken: string, rejectedIdentity: unknown): void;
}

export interface IdentityAccountScopeAdapter {
  capture(): AccountScope;
  isCurrent(scope: AccountScope): boolean;
}

export const createIdentityTransportAuthAdapter = (
  token: IdentityTokenAdapter,
  onUnauthorized: () => void,
  account: IdentityAccountScopeAdapter = {
    capture: captureAccountScope,
    isCurrent: isAccountScopeCurrent,
  }
): IdentityTransportAuthAdapter => ({
  getIdentity: account.capture,
  getToken: token.get,
  onUnauthorized: (rejectedToken, rejectedIdentity) => {
    if (
      token.get() === rejectedToken &&
      typeof rejectedIdentity === 'object' &&
      rejectedIdentity !== null &&
      account.isCurrent(rejectedIdentity as AccountScope)
    ) {
      onUnauthorized();
    }
  },
});

export const identityTransportAuthAdapter = createIdentityTransportAuthAdapter(
  browserIdentityTokenAdapter,
  handleUnauthorizedResponse
);

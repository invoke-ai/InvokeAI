import { captureAccountScope, isAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';

import type { IdentityTokenAdapter } from './core/tokenStorage';

import {
  browserIdentityTokenAdapter,
  isNewEpochForCurrentSession,
  isPasswordChangePending,
  waitForPasswordChange,
} from './core/tokenStorage';
import { handleUnauthorizedResponse } from './session';

export interface IdentityTransportAuthAdapter {
  getIdentity(): unknown;
  getToken(): string | null;
  onRefreshedToken(replacement: string, requestToken: string, requestIdentity: unknown): void;
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
  onRefreshedToken: (replacement, requestToken, requestIdentity) => {
    if (
      (token.get() === requestToken || isNewEpochForCurrentSession(requestToken, token.get(), replacement)) &&
      typeof requestIdentity === 'object' &&
      requestIdentity !== null &&
      account.isCurrent(requestIdentity as AccountScope)
    ) {
      token.set(replacement);
    }
  },
  onUnauthorized: (rejectedToken, rejectedIdentity) => {
    const isStillCurrent = (): boolean =>
      token.get() === rejectedToken &&
      typeof rejectedIdentity === 'object' &&
      rejectedIdentity !== null &&
      account.isCurrent(rejectedIdentity as AccountScope);

    if (!isStillCurrent()) {
      return;
    }

    if (isPasswordChangePending(rejectedToken)) {
      void waitForPasswordChange(rejectedToken, isStillCurrent).then(() => {
        if (isStillCurrent()) {
          onUnauthorized();
        }
      });
      return;
    }

    onUnauthorized();
  },
});

export const identityTransportAuthAdapter = createIdentityTransportAuthAdapter(
  browserIdentityTokenAdapter,
  handleUnauthorizedResponse
);

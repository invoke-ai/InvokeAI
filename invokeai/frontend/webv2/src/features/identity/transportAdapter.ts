import type { IdentityTokenAdapter } from './core/tokenStorage';

import { browserIdentityTokenAdapter } from './core/tokenStorage';
import { handleUnauthorizedResponse } from './session';

export interface IdentityTransportAuthAdapter {
  getToken(): string | null;
  onUnauthorized(): void;
}

export const createIdentityTransportAuthAdapter = (
  token: IdentityTokenAdapter,
  onUnauthorized: () => void
): IdentityTransportAuthAdapter => ({
  getToken: token.get,
  onUnauthorized,
});

export const identityTransportAuthAdapter = createIdentityTransportAuthAdapter(
  browserIdentityTokenAdapter,
  handleUnauthorizedResponse
);

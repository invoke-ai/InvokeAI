export interface AccountScope {
  /**
   * Stable backend identity for this authenticated lifetime. `null` represents
   * the boot/login lifetime, which is still epoch-scoped so its async work can
   * be invalidated before an account becomes active.
   */
  readonly accountId: string | null;
  /** Monotonically increasing identity lifetime, including same-user logins. */
  readonly epoch: number;
  /** Aborted synchronously when this identity lifetime becomes stale. */
  readonly signal: AbortSignal;
  /** Immutable suffix for account-owned browser storage. */
  readonly storageSuffix: string;
}

export interface AccountOwnedResource {
  /** Human-readable owner used by diagnostics and tests. */
  readonly name: string;
  /** Synchronously removes all data and cancels work owned by the old scope. */
  clear(): void;
}

export interface AccountLifecycle {
  /** Starts a fresh lifetime for `accountId`; same-account logins still rotate the epoch. */
  activate(accountId: string, storageSuffix?: string): AccountScope;
  /** Returns the immutable scope captured by work starting now. */
  capture(): AccountScope;
  /** Invalidates the current epoch before synchronously clearing registered resources. */
  invalidate(): AccountScope;
  /** True only while `scope` is the exact current identity lifetime. */
  isCurrent(scope: AccountScope): boolean;
  /** Registers an account-owned cache/runtime. Returns an unregister function. */
  register(resource: AccountOwnedResource): () => void;
}

export class AccountScopeExpiredError extends Error {
  constructor() {
    super('The account scope that owned this operation is no longer active.');
    this.name = 'AccountScopeExpiredError';
  }
}

const createScope = (
  accountId: string | null,
  epoch: number,
  signal: AbortSignal,
  storageSuffix: string
): AccountScope => Object.freeze({ accountId, epoch, signal, storageSuffix });

/**
 * Owns identity epochs and the account-resource registry behind one small
 * interface. Identity decides *when* a transition happens; the App composition
 * root supplies this lifecycle, and feature caches register independently so
 * lazy editor modules do not enter the Launchpad's eager dependency graph.
 */
export const createAccountLifecycle = (): AccountLifecycle => {
  let controller = new AbortController();
  let current = createScope(null, 0, controller.signal, '');
  const resources = new Map<string, { resource: AccountOwnedResource; token: number }>();
  let nextRegistrationToken = 1;

  const clearResources = (): void => {
    for (const { resource } of resources.values()) {
      try {
        resource.clear();
      } catch {
        // Privacy cleanup is fail-safe: one defective resource must not leave
        // every later cache live. Resource-specific tests own diagnostics.
      }
    }
  };

  const rotateScope = (accountId: string | null, storageSuffix: string): AccountScope => {
    const previousController = controller;

    controller = new AbortController();
    current = createScope(accountId, current.epoch + 1, controller.signal, storageSuffix);
    previousController.abort();

    return current;
  };

  const invalidate = (): AccountScope => {
    // Rotate first. Any completion racing a clear observes a stale epoch and
    // must become a no-op even if its transport cannot be aborted.
    rotateScope(null, '');
    clearResources();

    return current;
  };

  return {
    activate(accountId, storageSuffix = '') {
      if (!accountId) {
        throw new Error('An active account scope requires a non-empty account id.');
      }

      rotateScope(accountId, storageSuffix);
      clearResources();

      return current;
    },
    capture: () => current,
    invalidate,
    isCurrent: (scope) => scope === current,
    register(resource) {
      const registration = { resource, token: nextRegistrationToken };

      nextRegistrationToken += 1;
      resources.set(resource.name, registration);

      return () => {
        if (resources.get(resource.name)?.token === registration.token) {
          resources.delete(resource.name);
        }
      };
    },
  };
};

/** App-composed process lifecycle shared by account-owned feature modules. */
export const accountLifecycle = createAccountLifecycle();

export const captureAccountScope = (): AccountScope => accountLifecycle.capture();

export const isAccountScopeCurrent = (scope: AccountScope): boolean => accountLifecycle.isCurrent(scope);

export const assertAccountScopeCurrent = (scope: AccountScope): void => {
  if (!isAccountScopeCurrent(scope)) {
    throw new AccountScopeExpiredError();
  }
};

export const registerAccountOwnedResource = (resource: AccountOwnedResource): (() => void) =>
  accountLifecycle.register(resource);

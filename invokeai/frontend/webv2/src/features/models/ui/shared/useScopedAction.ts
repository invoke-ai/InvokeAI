import { type AccountScope, captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useCallback, useState } from 'react';

export const getErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

/**
 * Busy-tracked wrapper for account-scoped async UI actions. Owns the
 * boilerplate every mutation handler otherwise repeats: capture the account
 * scope, flip the busy flag, drop results that settle after an account
 * switch, and route failures to the caller's error sink.
 *
 * The action still calls `assertAccountScopeCurrent(owner)` itself before
 * patching stores on success — the throw lands here and is swallowed when the
 * scope is gone.
 */
export const useScopedAction = () => {
  const [isBusy, setIsBusy] = useState(false);

  const run = useCallback(
    async (
      action: (owner: AccountScope) => Promise<void>,
      onError?: (message: string, error: unknown) => void
    ): Promise<void> => {
      const owner = captureAccountScope();

      setIsBusy(true);

      try {
        await action(owner);
      } catch (error) {
        if (isAccountScopeCurrent(owner)) {
          onError?.(getErrorMessage(error), error);
        }
      } finally {
        if (isAccountScopeCurrent(owner)) {
          setIsBusy(false);
        }
      }
    },
    []
  );

  return { isBusy, run };
};

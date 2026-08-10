import { type AccountScope, captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useCallback, useRef, useState } from 'react';

export const getErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

/**
 * Busy-tracked wrapper for account-scoped async UI actions. Owns the
 * boilerplate every mutation handler otherwise repeats: capture the account
 * scope, flip the busy flag, drop results that settle after an account
 * switch, and route failures to the caller's error sink.
 *
 * `run` resolves `true` only when the action completed in the current account
 * scope, so callers can gate follow-up UI (navigation, closing a pane) on
 * success. While an action is in flight, further `run` calls resolve `false`
 * without invoking their action — keyboard paths that bypass a button's
 * `loading` state cannot double-submit.
 *
 * The action still calls `assertAccountScopeCurrent(owner)` itself before
 * patching stores on success — the throw lands here and is swallowed when the
 * scope is gone.
 */
export const useScopedAction = () => {
  const [isBusy, setIsBusy] = useState(false);
  const isBusyRef = useRef(false);

  const run = useCallback(
    async (
      action: (owner: AccountScope) => Promise<void>,
      onError?: (message: string, error: unknown) => void
    ): Promise<boolean> => {
      if (isBusyRef.current) {
        return false;
      }

      const owner = captureAccountScope();

      isBusyRef.current = true;
      setIsBusy(true);

      try {
        await action(owner);
        return isAccountScopeCurrent(owner);
      } catch (error) {
        if (isAccountScopeCurrent(owner)) {
          onError?.(getErrorMessage(error), error);
        }
        return false;
      } finally {
        isBusyRef.current = false;
        if (isAccountScopeCurrent(owner)) {
          setIsBusy(false);
        }
      }
    },
    []
  );

  return { isBusy, run };
};

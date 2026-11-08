import { type AccountScope, captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useCallback, useRef, useState } from 'react';

export const getErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

/**
 * Reject overlapping runs; return true only for current-account success. Actions must assertAccountScopeCurrent
 * before store writes; the wrapper cannot fence writes inside them.
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

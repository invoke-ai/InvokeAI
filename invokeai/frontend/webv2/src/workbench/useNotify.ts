import { useMemo } from 'react';

import type { WorkbenchNotificationKind } from './types';
import { useWorkbench } from './WorkbenchContext';

/**
 * Shell notification helper: `recordNotice` dispatch wrapped in one hook so
 * widgets never hand-roll the same notifyError/notifySuccess plumbing.
 */
export interface Notify {
  success: (title: string, message?: string) => void;
  error: (title: string, message?: string) => void;
  info: (title: string, message?: string) => void;
}

export const useNotify = (): Notify => {
  const { dispatch } = useWorkbench();

  return useMemo(() => {
    const record =
      (kind: WorkbenchNotificationKind) =>
      (title: string, message?: string): void => {
        dispatch({ kind, message, title, type: 'recordNotice' });
      };

    return { error: record('error'), info: record('info'), success: record('success') };
  }, [dispatch]);
};

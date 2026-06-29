import { toaster } from '@workbench/components/ui';
import { useMemo } from 'react';

import type { WorkbenchNotificationKind } from './types';

import { useOptionalWorkbenchDispatch } from './WorkbenchContext';

/**
 * Notification helper: records into the workbench shell when present, and
 * falls back to global toasts for project-independent surfaces like Home.
 */
export interface Notify {
  success: (title: string, message?: string) => void;
  error: (title: string, message?: string) => void;
  info: (title: string, message?: string) => void;
}

const notificationToastType: Record<WorkbenchNotificationKind, 'error' | 'info' | 'success'> = {
  error: 'error',
  info: 'info',
  success: 'success',
};

export const useNotify = (): Notify => {
  const dispatch = useOptionalWorkbenchDispatch();

  return useMemo(() => {
    const record =
      (kind: WorkbenchNotificationKind) =>
      (title: string, message?: string): void => {
        if (dispatch) {
          dispatch({ kind, message, title, type: 'recordNotice' });
        } else {
          toaster.create({ description: message, title, type: notificationToastType[kind] });
        }
      };

    return { error: record('error'), info: record('info'), success: record('success') };
  }, [dispatch]);
};

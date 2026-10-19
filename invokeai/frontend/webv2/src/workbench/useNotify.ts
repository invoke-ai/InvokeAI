import type { WorkbenchNotificationKind } from '@workbench/projectContracts';

import { toaster } from '@platform/ui';
import { useMemo } from 'react';

import { useOptionalWorkbenchCommands } from './WorkbenchContext';

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
  const commands = useOptionalWorkbenchCommands();

  return useMemo(() => {
    const record =
      (kind: WorkbenchNotificationKind) =>
      (title: string, message?: string): void => {
        if (commands) {
          commands.notifications.add({ kind, message, title });
        } else {
          toaster.create({ description: message, title, type: notificationToastType[kind] });
        }
      };

    return { error: record('error'), info: record('info'), success: record('success') };
  }, [commands]);
};

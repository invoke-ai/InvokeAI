import type { WorkbenchNotificationAction, WorkbenchNotificationKind } from '@workbench/projectContracts';

import { captureAccountScope } from '@platform/state/accountLifecycle';
import { ACTION_TOAST_DURATION_MS, createActionToast, toaster } from '@platform/ui/toaster';
import i18next from 'i18next';
import { useMemo } from 'react';

import {
  getNotificationUndoWithdrawal,
  holdNotificationUndo,
  registerNotificationUndo,
  releaseNotificationUndo,
  runNotificationUndo,
} from './shell/notifications/undoRegistry';
import { useOptionalWorkbenchCommands } from './WorkbenchContext';

export interface NotifyOptions {
  /** Toast buttons as plain descriptors (see `WorkbenchNotificationAction`); only the workbench can run them. */
  actions?: readonly WorkbenchNotificationAction[];
  /** The toast stays until dismissed: for failures the user must not miss. It still closes with the account. */
  persistent?: boolean;
}

export interface UndoableNotice {
  message?: string;
  /**
   * Reverts the action the notice reports. Offered once, while the toast shows, for this account only. The toast
   * outlives tab switches, so the closure must capture its target's ids (never "the active project") and do nothing,
   * or report, when the target is gone.
   */
  onUndo: () => void;
  /** The project the action changed: Undo is withdrawn once that project is no longer open. */
  projectId?: string;
  /**
   * What the undo reverts, such as one project's Generate settings. A newer undo with the same key closes this one's
   * toast: undoing an older step after a newer one would leave a mix of both.
   */
  supersedes?: string;
  /**
   * Toast only, never recorded in the notification center: for undoing a change to the center itself, which a record
   * of it would undo in turn (clearing the list would leave the clear's own notice in it).
   */
  transient?: boolean;
}

/**
 * Notification helper: records into the workbench shell when present, and
 * falls back to global toasts for project-independent surfaces like Home.
 */
export interface Notify {
  success: (title: string, message?: string, options?: NotifyOptions) => void;
  error: (title: string, message?: string, options?: NotifyOptions) => void;
  info: (title: string, message?: string, options?: NotifyOptions) => void;
  /** Reports an action that already happened, with an Undo button: the house style for reversible actions. */
  undoable: (title: string, notice: UndoableNotice) => void;
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
      (title: string, message?: string, options: NotifyOptions = {}): void => {
        if (commands) {
          commands.notifications.add({ ...options, kind, message, title });
        } else if (options.persistent) {
          createActionToast({
            actions: [],
            description: message,
            duration: Number.POSITIVE_INFINITY,
            signal: captureAccountScope().signal,
            title,
            type: notificationToastType[kind],
          });
        } else {
          toaster.create({ description: message, title, type: notificationToastType[kind] });
        }
      };

    const undoable = (title: string, { message, onUndo, projectId, supersedes, transient }: UndoableNotice): void => {
      const undoId = registerNotificationUndo(onUndo, supersedes);

      if (commands && !transient) {
        // The center keeps the notice as data; the closure waits in the registry until its toast closes.
        commands.notifications.add({
          actions: [{ kind: 'undo', undoId, ...(projectId ? { projectId } : {}) }],
          kind: 'success',
          message,
          title,
        });
        return;
      }

      const withdrawal = getNotificationUndoWithdrawal(undoId);

      createActionToast({
        // Read at call time: only this path needs a translation, and the hook must not suspend or change identity.
        actions: [{ label: i18next.t('common.undo'), onClick: () => runNotificationUndo(undoId) }],
        description: message,
        duration: ACTION_TOAST_DURATION_MS,
        onStatusChange: ({ status }) => {
          if (status === 'unmounted') {
            releaseNotificationUndo(undoId);
          }
        },
        signal: withdrawal ? AbortSignal.any([captureAccountScope().signal, withdrawal]) : captureAccountScope().signal,
        title,
        type: 'success',
      });
      holdNotificationUndo(undoId);
    };

    return { error: record('error'), info: record('info'), success: record('success'), undoable };
  }, [commands]);
};

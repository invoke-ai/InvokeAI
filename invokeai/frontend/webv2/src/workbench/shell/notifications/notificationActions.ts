import type { ToastAction } from '@platform/ui/toaster';
import type { WorkbenchNotification, WorkbenchNotificationAction } from '@workbench/projectContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';
import type { TFunction } from 'i18next';

import {
  getNotificationUndoWithdrawal,
  hasNotificationUndo,
  holdNotificationUndo,
  releaseNotificationUndo,
  retainNotificationUndos,
  runNotificationUndo,
  withdrawNotificationUndo,
} from './undoRegistry';

/** What the editor lends action handlers. Read when a toast opens (labels) and when its button is pressed. */
export interface NotificationActionContext {
  /** The widget's label while this editor can show it; null leaves its action out. */
  getWidgetLabel: (widgetId: WidgetTypeId) => string | null;
  /** Whether the project is still open in this editor; an undo scoped to a closed project is not offered. */
  isProjectOpen: (projectId: string) => boolean;
  /** Brings the widget forward where the layout has it, otherwise opens it, and moves focus there. */
  openWidget: (widgetId: WidgetTypeId) => void;
  /** Opens the queue at a backend item. */
  revealQueueItem: (itemId: number) => void;
  /** Whether a notice may still offer Retry for these backend items: not once they were retried from one. */
  canRetryQueueItem: (itemIds: readonly number[]) => boolean;
  /** Re-runs a run's failed or canceled backend items in one request, at most once, and reports how that went. */
  retryQueueItem: (action: { canvasProjectId?: string; itemIds: readonly number[] }) => void;
  t: TFunction;
}

type NotificationActionKind = WorkbenchNotificationAction['kind'];

interface NotificationActionHandler<Action extends WorkbenchNotificationAction> {
  /** The button label, or null when the action cannot be offered now. */
  label: (action: Action, context: NotificationActionContext) => string | null;
  run: (action: Action, context: NotificationActionContext) => void;
  /** Keeps what the action needs while a toast shows it, even once its record has left the list. */
  hold?: (action: Action) => void;
  /** Ends the hold of a toast that has closed; what no listed notification still offers goes then. */
  release?: (action: Action) => void;
  /** Aborts when the action is withdrawn for good; a toast offering it closes then. */
  withdrawal?: (action: Action) => AbortSignal | undefined;
}

/** One handler per descriptor kind; the mapped type makes a new kind a compile error until it has one. */
const handlers: {
  [Kind in NotificationActionKind]: NotificationActionHandler<Extract<WorkbenchNotificationAction, { kind: Kind }>>;
} = {
  'open-widget': {
    label: ({ widgetId }, { getWidgetLabel, t }) => {
      const widget = getWidgetLabel(widgetId);

      return widget === null ? null : t('notifications.actions.openWidget', { widget });
    },
    run: ({ widgetId }, { openWidget }) => openWidget(widgetId),
  },
  'reveal-queue-item': {
    label: (_action, { getWidgetLabel, t }) =>
      getWidgetLabel('queue') === null ? null : t('notifications.actions.showInQueue'),
    run: ({ itemId }, { revealQueueItem }) => revealQueueItem(itemId),
  },
  'retry-queue-item': {
    label: ({ itemIds }, { canRetryQueueItem, t }) => (canRetryQueueItem(itemIds) ? t('common.retry') : null),
    run: (action, { retryQueueItem }) => retryQueueItem(action),
  },
  undo: {
    hold: ({ undoId }) => holdNotificationUndo(undoId),
    label: ({ label, projectId, undoId }, { isProjectOpen, t }) =>
      hasNotificationUndo(undoId) && (projectId === undefined || isProjectOpen(projectId))
        ? (label ?? t('common.undo'))
        : null,
    release: ({ undoId }) => releaseNotificationUndo(undoId),
    withdrawal: ({ undoId }) => getNotificationUndoWithdrawal(undoId),
    run: ({ projectId, undoId }, { isProjectOpen }) => {
      // The toast can outlive its project's tab; undoing then would act on a project the user no longer sees.
      if (projectId !== undefined && !isProjectOpen(projectId)) {
        withdrawNotificationUndo(undoId);
        return;
      }

      runNotificationUndo(undoId);
    },
  },
};

/** Toast buttons for the actions that can be offered now, in their order. */
export const resolveNotificationActions = (
  actions: readonly WorkbenchNotificationAction[],
  context: NotificationActionContext
): ToastAction[] =>
  actions.flatMap((action) => {
    const handler = handlers[action.kind] as NotificationActionHandler<typeof action>;
    const label = handler.label(action, context);

    return label === null ? [] : [{ label, onClick: () => handler.run(action, context) }];
  });

/** Called when a toast shows `actions`. */
export const holdNotificationActions = (actions: readonly WorkbenchNotificationAction[]): void => {
  for (const action of actions) {
    (handlers[action.kind] as NotificationActionHandler<typeof action>).hold?.(action);
  }
};

/** Called when the toast that offered `actions` has closed. */
export const releaseNotificationActions = (actions: readonly WorkbenchNotificationAction[]): void => {
  for (const action of actions) {
    (handlers[action.kind] as NotificationActionHandler<typeof action>).release?.(action);
  }
};

/**
 * Keeps the undo entries the listed notifications still offer and frees the rest no toast holds: after the list
 * changed (a dismiss, Clear, a record pushed past the limit) or a project closed, whose undos are no longer offered.
 */
export const retainNotificationActions = (
  notifications: readonly WorkbenchNotification[],
  { isProjectOpen }: Pick<NotificationActionContext, 'isProjectOpen'>
): void => {
  const referencedIds = new Set<string>();

  for (const notification of notifications) {
    for (const action of notification.actions ?? []) {
      if (action.kind !== 'undo') {
        continue;
      }

      if (action.projectId !== undefined && !isProjectOpen(action.projectId)) {
        withdrawNotificationUndo(action.undoId);
      } else {
        referencedIds.add(action.undoId);
      }
    }
  }

  retainNotificationUndos(referencedIds);
};

/** Signals that close a toast offering `actions` once one of them is withdrawn (a superseded undo). */
export const getNotificationActionWithdrawals = (actions: readonly WorkbenchNotificationAction[]): AbortSignal[] =>
  actions.flatMap(
    (action) => (handlers[action.kind] as NotificationActionHandler<typeof action>).withdrawal?.(action) ?? []
  );

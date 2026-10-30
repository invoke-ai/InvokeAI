import type { ToastAction } from '@platform/ui/toaster';
import type { WorkbenchNotificationAction } from '@workbench/projectContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';
import type { TFunction } from 'i18next';

import {
  getNotificationUndoWithdrawal,
  hasNotificationUndo,
  holdNotificationUndo,
  releaseNotificationUndo,
  runNotificationUndo,
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
  /** Keeps what the action needs while a toast shows it. */
  hold?: (action: Action) => void;
  /** Frees what only a visible toast may still run, once that toast has closed. */
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
        releaseNotificationUndo(undoId);
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

/** Signals that close a toast offering `actions` once one of them is withdrawn (a superseded undo). */
export const getNotificationActionWithdrawals = (actions: readonly WorkbenchNotificationAction[]): AbortSignal[] =>
  actions.flatMap(
    (action) => (handlers[action.kind] as NotificationActionHandler<typeof action>).withdrawal?.(action) ?? []
  );

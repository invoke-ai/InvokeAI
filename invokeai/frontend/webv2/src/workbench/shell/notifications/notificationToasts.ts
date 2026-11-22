import type { WorkbenchNotification, WorkbenchNotificationKind } from '@workbench/projectContracts';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';
import type { TFunction } from 'i18next';

import { ACTION_TOAST_DURATION_MS, createActionToast, toaster } from '@platform/ui/toaster';

import {
  getNotificationActionWithdrawals,
  holdNotificationActions,
  releaseNotificationActions,
  resolveNotificationActions,
  retainNotificationActions,
  type NotificationActionContext,
} from './notificationActions';
import { getToastKey, replacesItsToast, shouldToastNotification } from './toastPolicy';

const notificationToastType: Record<WorkbenchNotificationKind, 'error' | 'info' | 'success'> = {
  error: 'error',
  info: 'info',
  success: 'success',
};

/** A notification's title and message in the current language; the English text stands in where no key exists. */
export const resolveNotificationText = (
  notification: WorkbenchNotification,
  t: TFunction
): { message: string | undefined; title: string } => ({
  message: notification.messageKey ? t(notification.messageKey, notification.messageValues) : notification.message,
  title: notification.titleKey ? t(notification.titleKey, notification.titleValues) : notification.title,
});

export interface NotificationToastSources {
  actionContext: NotificationActionContext;
  getNotifications: () => readonly WorkbenchNotification[];
  getPreferences: () => Pick<WorkbenchPreferences, 'notifyOnEnqueue'>;
  subscribe: (listener: () => void) => () => void;
}

/** The projects whose closing withdraws a listed undo. */
const getUndoProjectIds = (notifications: readonly WorkbenchNotification[]): string[] => [
  ...new Set(
    notifications.flatMap((notification) =>
      (notification.actions ?? []).flatMap((action) =>
        action.kind === 'undo' && action.projectId !== undefined ? [action.projectId] : []
      )
    )
  ),
];

/** Toasts notifications recorded after start; stopping dismisses toasts that would act on this editor. */
export const startNotificationToasts = ({
  actionContext,
  getNotifications,
  getPreferences,
  subscribe,
}: NotificationToastSources): (() => void) => {
  const lifetime = new AbortController();
  let notifications = getNotifications();
  // Only keys still listed can come back, so the set never outgrows the notification limit.
  let toastedKeys = new Set(notifications.map(getToastKey));
  // Records are immutable, so one listed before is the same object when Clear's Undo puts it back: it stays quiet.
  const listedRecords = new WeakSet<WorkbenchNotification>(notifications);
  let undoProjectIds: string[] = [];
  // The toast each replacing notification shows, by record id, so its repeat closes it first.
  const replaceableToasts = new Map<string, string>();

  const retain = (): void => {
    retainNotificationActions(notifications, actionContext);
    undoProjectIds = getUndoProjectIds(notifications).filter(actionContext.isProjectOpen);
  };

  const show = (notification: WorkbenchNotification): void => {
    const text = resolveNotificationText(notification, actionContext.t);
    const descriptors = notification.actions ?? [];
    const actions = resolveNotificationActions(descriptors, actionContext);
    const duration = notification.persistent
      ? Number.POSITIVE_INFINITY
      : actions.length > 0
        ? ACTION_TOAST_DURATION_MS
        : undefined;

    const shownId = replaceableToasts.get(notification.id);

    if (shownId !== undefined) {
      toaster.dismiss(shownId);
      replaceableToasts.delete(notification.id);
    }

    const toastId = createActionToast({
      actions,
      description: text.message,
      ...(duration === undefined ? {} : { duration }),
      ...(descriptors.length > 0
        ? {
            onStatusChange: ({ status }) => {
              if (status === 'unmounted') {
                releaseNotificationActions(descriptors);
              }
            },
          }
        : {}),
      ...(actions.length > 0 || notification.persistent
        ? { signal: AbortSignal.any([lifetime.signal, ...getNotificationActionWithdrawals(descriptors)]) }
        : {}),
      title: text.title,
      type: notificationToastType[notification.kind],
    });

    if (toastId !== null && replacesItsToast(notification)) {
      replaceableToasts.set(notification.id, toastId);
    }
    holdNotificationActions(descriptors);
  };

  // Entries recorded while no editor kept them in step: the listed ones count as referenced from now on.
  retain();
  const unsubscribe = subscribe(() => {
    const next = getNotifications();

    if (next === notifications) {
      if (undoProjectIds.some((projectId) => !actionContext.isProjectOpen(projectId))) {
        retain();
      }
      return;
    }

    notifications = next;
    retain();
    for (const notificationId of replaceableToasts.keys()) {
      if (!next.some(({ id }) => id === notificationId)) {
        replaceableToasts.delete(notificationId);
      }
    }
    const fresh = next
      .filter((notification) => !toastedKeys.has(getToastKey(notification)) && !listedRecords.has(notification))
      .reverse();
    toastedKeys = new Set(next.map(getToastKey));
    for (const notification of next) {
      listedRecords.add(notification);
    }
    const preferences = getPreferences();

    for (const notification of fresh) {
      // Left untoasted, the record alone keeps its actions, so the list can still offer them.
      if (!shouldToastNotification(notification, preferences)) {
        continue;
      }

      // Toast after the dispatch that recorded it has finished, never inside a React commit.
      queueMicrotask(() => {
        if (!lifetime.signal.aborted) {
          show(notification);
        }
      });
    }
  });

  return () => {
    unsubscribe();
    lifetime.abort();
  };
};

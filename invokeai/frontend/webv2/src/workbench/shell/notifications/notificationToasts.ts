import type { WorkbenchNotification, WorkbenchNotificationKind } from '@workbench/projectContracts';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';

import { ACTION_TOAST_DURATION_MS, createActionToast } from '@platform/ui/toaster';

import {
  holdNotificationActions,
  releaseNotificationActions,
  resolveNotificationActions,
  type NotificationActionContext,
} from './notificationActions';
import { getToastKey, shouldToastNotification } from './toastPolicy';

const notificationToastType: Record<WorkbenchNotificationKind, 'error' | 'info' | 'success'> = {
  error: 'error',
  info: 'info',
  success: 'success',
};

export interface NotificationToastSources {
  actionContext: NotificationActionContext;
  getNotifications: () => readonly WorkbenchNotification[];
  getPreferences: () => Pick<WorkbenchPreferences, 'notifyOnEnqueue'>;
  subscribe: (listener: () => void) => () => void;
}

/**
 * Toasts each notification recorded after start, oldest first. Notifications already present at start were seen
 * before (or belong to a session restore) and stay quiet. Stopping dismisses the toasts whose buttons would act on
 * this editor, and the persistent ones, which would otherwise outlive the account that saw them.
 */
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

  const show = (notification: WorkbenchNotification): void => {
    const { t } = actionContext;
    const descriptors = notification.actions ?? [];
    const actions = resolveNotificationActions(descriptors, actionContext);
    const duration = notification.persistent
      ? Number.POSITIVE_INFINITY
      : actions.length > 0
        ? ACTION_TOAST_DURATION_MS
        : undefined;

    createActionToast({
      actions,
      description: notification.messageKey ? t(notification.messageKey) : notification.message,
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
      ...(actions.length > 0 || notification.persistent ? { signal: lifetime.signal } : {}),
      title: notification.titleKey ? t(notification.titleKey) : notification.title,
      type: notificationToastType[notification.kind],
    });
    holdNotificationActions(descriptors);
  };

  const unsubscribe = subscribe(() => {
    const next = getNotifications();

    if (next === notifications) {
      return;
    }

    notifications = next;
    const fresh = next.filter((notification) => !toastedKeys.has(getToastKey(notification))).reverse();
    toastedKeys = new Set(next.map(getToastKey));
    const preferences = getPreferences();

    for (const notification of fresh) {
      if (!shouldToastNotification(notification, preferences)) {
        releaseNotificationActions(notification.actions ?? []);
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

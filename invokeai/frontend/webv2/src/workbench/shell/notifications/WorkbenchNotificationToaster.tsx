import type { WorkbenchNotificationKind } from '@workbench/projectContracts';

import { toaster } from '@platform/ui';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useEffect, useRef } from 'react';

const notificationToastType: Record<WorkbenchNotificationKind, 'error' | 'info' | 'success'> = {
  error: 'error',
  info: 'info',
  success: 'success',
};

export const WorkbenchNotificationToaster = () => {
  const notifications = useWorkbenchSelector((snapshot) => snapshot.notifications);
  const toastedNotificationIdsRef = useRef<Set<string> | null>(null);

  useEffect(() => {
    if (toastedNotificationIdsRef.current === null) {
      toastedNotificationIdsRef.current = new Set(notifications.map((notification) => notification.id));
      return;
    }

    for (const notification of [...notifications].reverse()) {
      if (toastedNotificationIdsRef.current.has(notification.id)) {
        continue;
      }

      toastedNotificationIdsRef.current.add(notification.id);
      queueMicrotask(() => {
        toaster.create({
          description: notification.message,
          title: notification.title,
          type: notificationToastType[notification.kind],
        });
      });
    }
  }, [notifications]);

  return null;
};

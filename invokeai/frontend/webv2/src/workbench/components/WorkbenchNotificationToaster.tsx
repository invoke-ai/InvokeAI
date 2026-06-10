import { useEffect, useRef } from 'react';

import type { WorkbenchNotificationKind } from '../types';
import { useWorkbench } from '../WorkbenchContext';
import { toaster } from './ui/toaster';

const notificationToastType: Record<WorkbenchNotificationKind, 'error' | 'info' | 'success'> = {
  error: 'error',
  info: 'info',
  success: 'success',
};

export const WorkbenchNotificationToaster = () => {
  const { state } = useWorkbench();
  const toastedNotificationIdsRef = useRef<Set<string> | null>(null);

  useEffect(() => {
    if (toastedNotificationIdsRef.current === null) {
      toastedNotificationIdsRef.current = new Set(state.notifications.map((notification) => notification.id));
      return;
    }

    for (const notification of [...state.notifications].reverse()) {
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
  }, [state.notifications]);

  return null;
};

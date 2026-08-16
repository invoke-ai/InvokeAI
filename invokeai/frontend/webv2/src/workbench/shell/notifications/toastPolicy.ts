import type { WorkbenchNotification } from '@workbench/projectContracts';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';

/** Whether a recorded notification should also surface as a toast. */
export const shouldToastNotification = (
  notification: WorkbenchNotification,
  prefs: Pick<WorkbenchPreferences, 'notifyOnEnqueue'>
): boolean => (notification.category === 'enqueue' ? prefs.notifyOnEnqueue : true);

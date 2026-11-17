import type { WorkbenchNotification } from '@workbench/projectContracts';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';

export const shouldToastNotification = (
  notification: WorkbenchNotification,
  prefs: Pick<WorkbenchPreferences, 'notifyOnEnqueue'>
): boolean => (notification.category === 'enqueue' ? prefs.notifyOnEnqueue : true);

/** Ambient repeats bump `occurrenceCount` without re-toasting; a user's run failing again re-toasts. */
export const getToastKey = (notification: WorkbenchNotification): string =>
  notification.category === 'run-outcome' ? `${notification.id}:${notification.occurrenceCount ?? 1}` : notification.id;

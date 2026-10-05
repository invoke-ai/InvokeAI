import type { WorkbenchNotification } from '@workbench/projectContracts';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';

/** Whether a recorded notification should also surface as a toast. */
export const shouldToastNotification = (
  notification: WorkbenchNotification,
  prefs: Pick<WorkbenchPreferences, 'notifyOnEnqueue'>
): boolean => (notification.category === 'enqueue' ? prefs.notifyOnEnqueue : true);

/**
 * Coalesced repeats bump `occurrenceCount` on the same id. Ambient errors (a retry failing the
 * same way every cycle) must not re-toast; a run the user submitted failing the same way again must.
 */
export const getToastKey = (notification: WorkbenchNotification): string =>
  notification.category === 'run-outcome' ? `${notification.id}:${notification.occurrenceCount ?? 1}` : notification.id;

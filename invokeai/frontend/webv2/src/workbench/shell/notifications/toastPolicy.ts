import type { WorkbenchNotification } from '@workbench/projectContracts';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';

export const shouldToastNotification = (
  notification: WorkbenchNotification,
  prefs: Pick<WorkbenchPreferences, 'notifyOnEnqueue'>
): boolean => (notification.category === 'enqueue' ? prefs.notifyOnEnqueue : true);

/**
 * Ambient repeats bump `occurrenceCount` without re-toasting; a user's run failing again re-toasts, and so does a
 * repeated open that found nothing to show, so each gesture gets an answer.
 */
export const getToastKey = (notification: WorkbenchNotification): string =>
  notification.category === 'run-outcome' || notification.category === 'workspace-offer'
    ? `${notification.id}:${notification.occurrenceCount ?? 1}`
    : notification.id;

/** A repeated offer replaces its own toast rather than stacking beside it. */
export const replacesItsToast = (notification: WorkbenchNotification): boolean =>
  notification.category === 'workspace-offer';

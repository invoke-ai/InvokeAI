import type { S } from 'services/api/types';

/**
 * The user-scoped view of the queue counts, for personal UI: the progress bar, the busy favicon,
 * and the invoke-button spinner. In multiuser mode the global counts include other users'
 * generations, so the per-user counts are preferred; a status that carries no per-user counts
 * (e.g. the sanitized statuses broadcast to non-owners) falls back to the global counts, which
 * are correct in single-user mode.
 *
 * This is the single place encoding that fallback — personal-UI consumers must not reimplement
 * it, or they drift and disagree about whether the user is busy.
 */
export const getUserScopedQueueCounts = (queue: S['SessionQueueStatus']): { pending: number; inProgress: number } => ({
  pending: queue.user_pending ?? queue.pending,
  inProgress: queue.user_in_progress ?? queue.in_progress,
});

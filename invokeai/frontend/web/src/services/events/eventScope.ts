import type { AppGetState } from 'app/store/store';
import { selectAuthToken, selectCurrentUser } from 'features/auth/store/authSlice';

/**
 * Who a socket event belongs to, relative to this client:
 *
 * - 'own': the current user's event. In single-user mode there is no authenticated user and every
 *   event is 'own'.
 * - 'foreign': another user's event, carrying that user's real user_id. Admin clients receive
 *   these via the "admin" socket room; regular users never do.
 * - 'sanitized': the redacted companion event the backend broadcasts to non-owner queue
 *   subscribers so their queue lists and badge counts stay in sync. Identified by the
 *   user_id="redacted" sentinel; identifiers and error fields are cleared. Only queue events
 *   (queue_item_status_changed, queue_cleared, queue_items_retried) have sanitized companions —
 *   invocation events are never sanitized.
 */
type EventScope = 'own' | 'foreign' | 'sanitized';

export const REDACTED_USER_ID = 'redacted';

/**
 * Classifies a socket event by ownership. Listeners use this to route events: only 'own' events
 * may drive personal UI state (progress, node execution states, gallery selection, toasts);
 * 'foreign' and 'sanitized' events are limited to cache invalidation at most.
 */
export const getEventScope = (getState: AppGetState, data: { user_id?: string | null }): EventScope => {
  if (data.user_id === REDACTED_USER_ID) {
    return 'sanitized';
  }
  const state = getState();
  const currentUser = selectCurrentUser(state);
  if (!currentUser) {
    // No authenticated user can mean two things:
    //
    // - Single-user mode: there is no auth token and auth.user never populates. Every event is
    //   the client's own.
    // - Multiuser mode before hydration: useSocketIO defers the socket connection until
    //   auth.user has hydrated from /me, so no events should arrive in this window. Should one
    //   arrive anyway, it cannot be attributed yet — classify it as foreign (cache invalidation
    //   at most) rather than let an event that may be another user's drive personal UI.
    //   Ownership-driven one-shot side effects (progress, node execution states, gallery
    //   auto-switch, the failure toast) never replay, which is exactly why the connection is
    //   deferred instead of events being buffered here.
    //
    // The token distinguishes the two: it is hydrated synchronously from localStorage, and
    // ProtectedRoute clears a stale token when the server reports single-user mode.
    return selectAuthToken(state) ? 'foreign' : 'own';
  }
  return data.user_id === currentUser.user_id ? 'own' : 'foreign';
};

import { useAppSelector } from 'app/store/storeHooks';
import { selectIsAuthenticated } from 'features/auth/store/authSlice';
import { notifyMediaCookieRefreshed } from 'features/auth/store/mediaCookieRefresh';
import { useEffect, useRef } from 'react';
import { useRefreshMediaCookieMutation } from 'services/api/endpoints/auth';

// Bounded backoff for transient failures (network hiccup, server 5xx during app
// load). Two retries is enough to ride out a restart; anything longer-lived needs a
// reload anyway.
const RETRY_DELAYS_MS = [2_000, 10_000];

let pauseRefreshHandler: (() => Promise<() => void>) | null = null;
type PendingRefresh = { promise: Promise<void>; abort: () => void };
const pendingRefreshes = new Set<PendingRefresh>();

export const abortAndWaitForPendingRefreshes = async (pending: Set<PendingRefresh>) => {
  for (const refresh of pending) {
    refresh.abort();
  }
  await Promise.all([...pending].map((refresh) => refresh.promise));
};

export const pauseMediaCookieRefreshForLogout = (): Promise<() => void> =>
  pauseRefreshHandler?.() ?? Promise.resolve(() => undefined);

/**
 * Self-heal the media cookie for restored sessions.
 *
 * Video playback authenticates via an HttpOnly cookie (media elements can't send
 * Authorization headers) that is only set at login. A session restored from
 * localStorage can hold a valid JWT without the cookie — every API call works but
 * each `<video>` request 401s and the player renders black with 0:00 duration.
 * Re-issuing the cookie from the Bearer token on app load closes that gap.
 *
 * Runs when an authenticated session exists (fresh logins get the cookie from the
 * login response too — the repeat is a harmless no-op). In single-user mode
 * `isAuthenticated` is never set, and media routes don't require the cookie anyway.
 *
 * Transient failures retry on a short bounded backoff — a network error or 5xx
 * during app load must not leave an otherwise-valid session unable to load media
 * until a full reload. A 401 is not retried: the session itself is expired, which
 * the global 401 handling already deals with, and retrying would loop.
 */
export const useMediaCookieRefresh = () => {
  const isAuthenticated = useAppSelector(selectIsAuthenticated);
  const [refreshMediaCookie] = useRefreshMediaCookieMutation();
  const hasSucceeded = useRef(false);

  useEffect(() => {
    if (!isAuthenticated || hasSucceeded.current) {
      return;
    }

    let canceled = false;
    let paused = false;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let nextAttemptIndex: number | null = null;

    const attempt = (attemptIndex: number) => {
      if (canceled || paused) {
        return;
      }
      nextAttemptIndex = null;
      const request = refreshMediaCookie();
      const refreshPromise = request
        .unwrap()
        .then(() => {
          if (canceled) {
            return;
          }
          hasSucceeded.current = true;
          notifyMediaCookieRefreshed();
        })
        .catch((error: unknown) => {
          if (canceled) {
            return;
          }
          const status = (error as { status?: unknown } | null)?.status;
          if (status === 401) {
            return;
          }
          const delay = RETRY_DELAYS_MS[attemptIndex];
          if (delay === undefined) {
            return;
          }
          nextAttemptIndex = attemptIndex + 1;
          if (!paused) {
            timeoutId = setTimeout(() => attempt(attemptIndex + 1), delay);
          }
        });
      const pendingRefresh = { promise: refreshPromise, abort: request.abort };
      pendingRefreshes.add(pendingRefresh);
      void refreshPromise.finally(() => pendingRefreshes.delete(pendingRefresh));
    };

    const pause = async () => {
      paused = true;
      if (timeoutId !== undefined) {
        clearTimeout(timeoutId);
        timeoutId = undefined;
      }
      await abortAndWaitForPendingRefreshes(pendingRefreshes);
      return () => {
        if (canceled || hasSucceeded.current) {
          return;
        }
        paused = false;
        if (nextAttemptIndex !== null) {
          attempt(nextAttemptIndex);
        }
      };
    };

    pauseRefreshHandler = pause;
    attempt(0);

    return () => {
      canceled = true;
      if (pauseRefreshHandler === pause) {
        pauseRefreshHandler = null;
      }
      if (timeoutId !== undefined) {
        clearTimeout(timeoutId);
      }
    };
  }, [isAuthenticated, refreshMediaCookie]);
};

import { useAppSelector } from 'app/store/storeHooks';
import { selectIsAuthenticated } from 'features/auth/store/authSlice';
import { notifyMediaCookieRefreshed } from 'features/auth/store/mediaCookieRefresh';
import { useEffect, useRef } from 'react';
import { useRefreshMediaCookieMutation } from 'services/api/endpoints/auth';

/**
 * Self-heal the media cookie for restored sessions.
 *
 * Video playback authenticates via an HttpOnly cookie (media elements can't send
 * Authorization headers) that is only set at login. A session restored from
 * localStorage can hold a valid JWT without the cookie — every API call works but
 * each `<video>` request 401s and the player renders black with 0:00 duration.
 * Re-issuing the cookie from the Bearer token on app load closes that gap.
 *
 * Fires once per app load when an authenticated session exists (fresh logins get
 * the cookie from the login response too — the repeat is a harmless no-op). In
 * single-user mode `isAuthenticated` is never set, and media routes don't require
 * the cookie anyway. Failures are ignored: a 401 here means the session itself is
 * expired, which the global 401 handling already deals with.
 */
export const useMediaCookieRefresh = () => {
  const isAuthenticated = useAppSelector(selectIsAuthenticated);
  const [refreshMediaCookie] = useRefreshMediaCookieMutation();
  const hasRefreshed = useRef(false);

  useEffect(() => {
    if (!isAuthenticated || hasRefreshed.current) {
      return;
    }
    hasRefreshed.current = true;
    void refreshMediaCookie()
      .unwrap()
      .then(notifyMediaCookieRefreshed)
      .catch(() => undefined);
  }, [isAuthenticated, refreshMediaCookie]);
};

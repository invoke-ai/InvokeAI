import { useEffect } from 'react';

import { socketHub } from './socketHub';

/**
 * Renders nothing. Mounted once above both the Launchpad and the editor: opens
 * the single backend socket for the authenticated session. Feature-specific
 * runtimes attach listeners where needed so this base runtime stays lightweight.
 *
 * It intentionally does NOT disconnect on unmount — that keeps it StrictMode
 * safe and lets the socket persist across Launchpad↔editor navigation. The
 * socket is torn down explicitly on logout/expiry (see `auth/session.ts`).
 */
export const SocketHubRuntime = () => {
  useEffect(() => {
    socketHub.connect();
  }, []);

  return null;
};

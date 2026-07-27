import { useMountEffect } from '@platform/react/useMountEffect';
import { socketHub } from '@platform/transport/socketHub';

/**
 * Renders nothing. Mounted once above both the Launchpad and the editor: opens
 * the single backend socket for the authenticated session. Feature-specific
 * runtimes attach listeners where needed so this base runtime stays lightweight.
 *
 * It intentionally does NOT disconnect on unmount — that keeps it StrictMode
 * safe and lets the socket persist across Launchpad↔editor navigation. The
 * App-composed account lifecycle tears it down before an identity epoch changes.
 */
export const SocketHubRuntime = () => {
  useMountEffect(() => {
    socketHub.connect();
  });

  return null;
};

import { useMountEffect } from '@platform/react/useMountEffect';
import { socketHub } from '@platform/transport/socketHub';

/**
 * Connect once per authenticated account. Account lifecycle owns teardown; unmount cleanup would disconnect during
 * StrictMode or route changes.
 */
export const SocketHubRuntime = () => {
  useMountEffect(() => {
    socketHub.connect();
  });

  return null;
};

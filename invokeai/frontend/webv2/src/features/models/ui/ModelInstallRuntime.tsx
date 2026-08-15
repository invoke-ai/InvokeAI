import {
  getInstallsSnapshot,
  handleModelInstallSocketEvent,
  MODEL_INSTALL_SOCKET_EVENTS,
  refreshInstalls,
} from '@features/models/data/installsStore';
import { getModelsSnapshot, refreshModels } from '@features/models/data/modelsStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getConnectionStatus, subscribeConnection } from '@platform/transport/connectionStore';
import { socketHub } from '@platform/transport/socketHub';

import { useInstallOutcomeToasts } from './useInstallOutcomeToasts';

/**
 * App-wide install runtime: install socket events, reconnect refresh, and
 * outcome toasts. Mounted once in the authenticated layout so installs
 * progress, refresh the library, and announce completion no matter which
 * surface queued them. Exactly one instance may exist — the outcome toasts
 * keep per-instance seen state.
 */
export const ModelInstallRuntime = () => {
  useMountEffect(() => {
    const owner = captureAccountScope();
    const detachers = MODEL_INSTALL_SOCKET_EVENTS.map((event) =>
      socketHub.on(event, (payload) => handleModelInstallSocketEvent(event, payload, owner))
    );

    return () => {
      for (const detach of detachers) {
        detach();
      }
    };
  });

  useMountEffect(() => {
    const owner = captureAccountScope();
    const refreshOnConnect = () => {
      if (!isAccountScopeCurrent(owner) || getConnectionStatus().status !== 'connected') {
        return;
      }

      // Only revalidate stores something already read — an app-wide mount
      // must not fetch model data for sessions that never open a models
      // surface. An install finishing while a store is idle still lands via
      // the socket handler's scheduled refresh.
      if (getModelsSnapshot().status !== 'idle') {
        void refreshModels();
      }

      if (getInstallsSnapshot().status !== 'idle') {
        void refreshInstalls();
      }
    };

    refreshOnConnect();

    return subscribeConnection(refreshOnConnect);
  });

  useInstallOutcomeToasts();

  return null;
};

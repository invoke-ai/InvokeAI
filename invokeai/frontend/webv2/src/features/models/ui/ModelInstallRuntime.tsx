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
 * Mount exactly one authenticated install runtime; per-instance toast history would otherwise duplicate completion
 * announcements.
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

      // Reconnect only refreshes previously read stores; install completions separately schedule refreshes for
      // idle stores.
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

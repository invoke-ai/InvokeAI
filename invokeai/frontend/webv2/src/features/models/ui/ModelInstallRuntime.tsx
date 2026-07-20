import {
  handleModelInstallSocketEvent,
  MODEL_INSTALL_SOCKET_EVENTS,
  refreshInstalls,
} from '@features/models/data/installsStore';
import { refreshModels } from '@features/models/data/modelsStore';
import { getConnectionStatus, subscribeConnection } from '@platform/transport/connectionStore';
import { socketHub } from '@platform/transport/socketHub';
import { useEffect } from 'react';

import { useInstallOutcomeToasts } from './useInstallOutcomeToasts';

/** Admin model-manager runtime: install socket events, reconnect refresh, and outcome toasts. */
export const ModelInstallRuntime = () => {
  useEffect(() => {
    const detachers = MODEL_INSTALL_SOCKET_EVENTS.map((event) =>
      socketHub.on(event, (payload) => handleModelInstallSocketEvent(event, payload))
    );

    return () => {
      for (const detach of detachers) {
        detach();
      }
    };
  }, []);

  useEffect(() => {
    const refreshOnConnect = () => {
      if (getConnectionStatus().status === 'connected') {
        void refreshModels();
        void refreshInstalls();
      }
    };

    refreshOnConnect();

    return subscribeConnection(refreshOnConnect);
  }, []);

  useInstallOutcomeToasts();

  return null;
};

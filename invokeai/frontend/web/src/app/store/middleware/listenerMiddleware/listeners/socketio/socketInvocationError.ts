import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { deepClone } from 'common/util/deepClone';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useExecutionState';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { toast } from 'features/toast/toast';
import ToastWithSessionRefDescription from 'features/toast/ToastWithSessionRefDescription';
import { t } from 'i18next';
import { startCase } from 'lodash-es';
import { socketInvocationError } from 'services/events/actions';

const log = logger('socketio');

const getTitle = (errorType: string) => {
  if (errorType === 'OutOfMemoryError') {
    return t('toast.outOfMemoryError');
  }
  return t('toast.serverError');
};

const getDescription = (errorType: string, sessionId: string, isLocal?: boolean) => {
  if (!isLocal) {
    if (errorType === 'OutOfMemoryError') {
      return ToastWithSessionRefDescription({
        message: t('toast.outOfMemoryDescription'),
        sessionId,
      });
    }
    return ToastWithSessionRefDescription({
      message: errorType,
      sessionId,
    });
  }
  return errorType;
};

export const addInvocationErrorEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketInvocationError,
    effect: (action, { getState }) => {
      log.error(action.payload, `Invocation error (${action.payload.data.node.type})`);
      const { source_node_id, error_type, error_message, error_traceback, graph_execution_state_id } =
        action.payload.data;
      const nes = deepClone($nodeExecutionStates.get()[source_node_id]);
      if (nes) {
        nes.status = zNodeStatus.enum.FAILED;
        nes.progress = null;
        nes.progressImage = null;

        nes.error = {
          error_type,
          error_message,
          error_traceback,
        };
        upsertExecutionState(nes.nodeId, nes);
      }

      const errorType = startCase(error_type);
      const sessionId = graph_execution_state_id;
      const { isLocal } = getState().config;

      toast({
        id: `INVOCATION_ERROR_${errorType}`,
        title: getTitle(errorType),
        status: 'error',
        duration: null,
        description: getDescription(errorType, sessionId, isLocal),
        updateDescription: isLocal ? true : false,
      });
    },
  });
};

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

export const addInvocationErrorEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketInvocationError,
    effect: (action) => {
      log.error(action.payload, `Invocation error (${action.payload.data.node.type})`);
      const { source_node_id, error_type } = action.payload.data;
      const nes = deepClone($nodeExecutionStates.get()[source_node_id]);
      if (nes) {
        nes.status = zNodeStatus.enum.FAILED;
        nes.error = action.payload.data.error;
        nes.progress = null;
        nes.progressImage = null;
        upsertExecutionState(nes.nodeId, nes);
      }
      const errorType = startCase(action.payload.data.error_type);
      const sessionId = action.payload.data.graph_execution_state_id;

      if (error_type === 'OutOfMemoryError') {
        toast({
          id: 'INVOCATION_ERROR',
          title: t('toast.outOfMemoryError'),
          status: 'error',
          duration: null,
          description: ToastWithSessionRefDescription({
            message: t('toast.outOfMemoryDescription'),
            sessionId,
          }),
        });
      } else {
        toast({
          id: `INVOCATION_ERROR_${errorType}`,
          title: t('toast.serverError'),
          status: 'error',
          duration: null,
          description: ToastWithSessionRefDescription({
            message: errorType,
            sessionId,
          }),
        });
      }
    },
  });
};

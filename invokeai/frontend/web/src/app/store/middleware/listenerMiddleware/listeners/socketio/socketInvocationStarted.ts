import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { deepClone } from 'common/util/deepClone';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useExecutionState';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { socketInvocationStarted } from 'services/events/actions';

const log = logger('socketio');

export const addInvocationStartedEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketInvocationStarted,
    effect: (action) => {
      log.debug(action.payload, `Invocation started (${action.payload.data.node.type})`);
      const { source_node_id } = action.payload.data;
      const nes = deepClone($nodeExecutionStates.get()[source_node_id]);
      if (nes) {
        nes.status = zNodeStatus.enum.IN_PROGRESS;
        upsertExecutionState(nes.nodeId, nes);
      }
    },
  });
};

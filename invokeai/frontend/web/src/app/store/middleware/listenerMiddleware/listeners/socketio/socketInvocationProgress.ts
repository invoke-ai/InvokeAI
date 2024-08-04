import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { deepClone } from 'common/util/deepClone';
import { parseify } from 'common/util/serialize';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useExecutionState';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { socketInvocationProgress } from 'services/events/actions';

const log = logger('socketio');

export const addInvocationProgressEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketInvocationProgress,
    effect: (action) => {
      log.trace(parseify(action.payload), `Generator progress`);
      const { invocation_source_id, percentage, image } = action.payload.data;
      const nes = deepClone($nodeExecutionStates.get()[invocation_source_id]);
      if (nes) {
        nes.status = zNodeStatus.enum.IN_PROGRESS;
        nes.progress = percentage;
        nes.progressImage = image ?? null;
        upsertExecutionState(nes.nodeId, nes);
      }
    },
  });
};

import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { deepClone } from 'common/util/deepClone';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useExecutionState';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { socketGeneratorProgress } from 'services/events/actions';

const log = logger('socketio');

export const addGeneratorProgressEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketGeneratorProgress,
    effect: (action) => {
      log.trace(action.payload, `Generator progress`);
      const { invocation_source_id, step, total_steps, progress_image } = action.payload.data;
      const nes = deepClone($nodeExecutionStates.get()[invocation_source_id]);
      if (nes) {
        nes.status = zNodeStatus.enum.IN_PROGRESS;
        nes.progress = (step + 1) / total_steps;
        nes.progressImage = progress_image ?? null;
        upsertExecutionState(nes.nodeId, nes);
      }
    },
  });
};

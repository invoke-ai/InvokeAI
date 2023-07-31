import { logger } from 'app/logging/logger';
import { userInvoked } from 'app/store/actions';
import { parseify } from 'common/util/serialize';
import { imageToImageGraphBuilt } from 'features/nodes/store/actions';
import { buildLinearImageToImageGraph } from 'features/nodes/util/graphBuilders/buildLinearImageToImageGraph';
import { buildLinearSDXLImageToImageGraph } from 'features/nodes/util/graphBuilders/buildLinearSDXLImageToImageGraph';
import { sessionReadyToInvoke } from 'features/system/store/actions';
import { sessionCreated } from 'services/api/thunks/session';
import { startAppListening } from '..';

export const addUserInvokedImageToImageListener = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof userInvoked> =>
      userInvoked.match(action) && action.payload === 'img2img',
    effect: async (action, { getState, dispatch, take }) => {
      const log = logger('session');
      const state = getState();
      const model = state.generation.model;

      let graph;

      if (model && model.base_model === 'sdxl') {
        graph = buildLinearSDXLImageToImageGraph(state);
      } else {
        graph = buildLinearImageToImageGraph(state);
      }

      dispatch(imageToImageGraphBuilt(graph));
      log.debug({ graph: parseify(graph) }, 'Image to Image graph built');

      dispatch(sessionCreated({ graph }));

      await take(sessionCreated.fulfilled.match);

      dispatch(sessionReadyToInvoke());
    },
  });
};

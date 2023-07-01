import { startAppListening } from '..';
import { sessionCreated } from 'services/api/thunks/session';
import { log } from 'app/logging/useLogger';
import { imageToImageGraphBuilt } from 'features/nodes/store/actions';
import { userInvoked } from 'app/store/actions';
import { sessionReadyToInvoke } from 'features/system/store/actions';
import { buildLinearImageToImageGraph } from 'features/nodes/util/graphBuilders/buildLinearImageToImageGraph';

const moduleLog = log.child({ namespace: 'invoke' });

export const addUserInvokedImageToImageListener = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof userInvoked> =>
      userInvoked.match(action) && action.payload === 'img2img',
    effect: async (action, { getState, dispatch, take }) => {
      const state = getState();

      const graph = buildLinearImageToImageGraph(state);
      dispatch(imageToImageGraphBuilt(graph));
      moduleLog.debug({ data: graph }, 'Image to Image graph built');

      dispatch(sessionCreated({ graph }));

      await take(sessionCreated.fulfilled.match);

      dispatch(sessionReadyToInvoke());
    },
  });
};

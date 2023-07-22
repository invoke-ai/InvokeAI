import { createAction } from '@reduxjs/toolkit';
import { buildAdHocUpscaleGraph } from 'features/nodes/util/graphBuilders/buildAdHocUpscaleGraph';
import { sessionReadyToInvoke } from 'features/system/store/actions';
import { sessionCreated } from 'services/api/thunks/session';
import { startAppListening } from '..';

export const upscaleRequested = createAction<{ image_name: string }>(
  `upscale/upscaleRequested`
);

export const addUpscaleRequestedListener = () => {
  startAppListening({
    actionCreator: upscaleRequested,
    effect: async (action, { dispatch, getState, take }) => {
      const { image_name } = action.payload;
      const { esrganModelName } = getState().postprocessing;

      const graph = buildAdHocUpscaleGraph({
        image_name,
        esrganModelName,
      });

      // Create a session to run the graph & wait til it's ready to invoke
      dispatch(sessionCreated({ graph }));

      await take(sessionCreated.fulfilled.match);

      dispatch(sessionReadyToInvoke());
    },
  });
};

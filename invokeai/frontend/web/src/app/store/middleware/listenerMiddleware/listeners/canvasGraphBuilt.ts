import { canvasGraphBuilt } from 'features/nodes/store/actions';
import { startAppListening } from '..';
import {
  canvasSessionIdChanged,
  stagingAreaInitialized,
} from 'features/canvas/store/canvasSlice';
import { sessionInvoked } from 'services/thunks/session';

export const addCanvasGraphBuiltListener = () =>
  startAppListening({
    actionCreator: canvasGraphBuilt,
    effect: async (action, { dispatch, getState, take }) => {
      const [{ meta }] = await take(sessionInvoked.fulfilled.match);
      const { sessionId } = meta.arg;
      const state = getState();

      if (!state.canvas.layerState.stagingArea.boundingBox) {
        dispatch(
          stagingAreaInitialized({
            sessionId,
            boundingBox: {
              ...state.canvas.boundingBoxCoordinates,
              ...state.canvas.boundingBoxDimensions,
            },
          })
        );
      }

      dispatch(canvasSessionIdChanged(sessionId));
    },
  });

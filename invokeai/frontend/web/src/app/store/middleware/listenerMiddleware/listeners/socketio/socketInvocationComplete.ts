import { addImageToStagingArea } from 'features/canvas/store/canvasSlice';
import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import {
  appSocketInvocationComplete,
  socketInvocationComplete,
} from 'services/events/actions';
import { imageMetadataReceived } from 'services/thunks/image';
import { sessionCanceled } from 'services/thunks/session';
import { isImageOutput } from 'services/types/guards';
import { progressImageSet } from 'features/system/store/systemSlice';

const moduleLog = log.child({ namespace: 'socketio' });
const nodeDenylist = ['dataURL_image'];

export const addInvocationCompleteEventListener = () => {
  startAppListening({
    actionCreator: socketInvocationComplete,
    effect: async (action, { dispatch, getState, take }) => {
      moduleLog.debug(
        action.payload,
        `Invocation complete (${action.payload.data.node.type})`
      );

      const sessionId = action.payload.data.graph_execution_state_id;

      const { cancelType, isCancelScheduled } = getState().system;

      // Handle scheduled cancelation
      if (cancelType === 'scheduled' && isCancelScheduled) {
        dispatch(sessionCanceled({ sessionId }));
      }

      const { data } = action.payload;
      const { result, node, graph_execution_state_id } = data;

      // This complete event has an associated image output
      if (isImageOutput(result) && !nodeDenylist.includes(node.type)) {
        const { image_name, image_origin } = result.image;

        // Get its metadata
        dispatch(
          imageMetadataReceived({
            imageName: image_name,
            imageOrigin: image_origin,
          })
        );

        const [{ payload: imageDTO }] = await take(
          imageMetadataReceived.fulfilled.match
        );

        // Handle canvas image
        if (
          graph_execution_state_id ===
          getState().canvas.layerState.stagingArea.sessionId
        ) {
          dispatch(addImageToStagingArea(imageDTO));
        }

        dispatch(progressImageSet(null));
      }
      // pass along the socket event as an application action
      dispatch(appSocketInvocationComplete(action.payload));
    },
  });
};

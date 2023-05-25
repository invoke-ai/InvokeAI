import { addImageToStagingArea } from 'features/canvas/store/canvasSlice';
import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import { invocationComplete } from 'services/events/actions';
import {
  imageMetadataReceived,
  imageUrlsReceived,
} from 'services/thunks/image';
import { sessionCanceled } from 'services/thunks/session';
import { isImageOutput } from 'services/types/guards';

const moduleLog = log.child({ namespace: 'socketio' });
const nodeDenylist = ['dataURL_image'];

export const addInvocationCompleteListener = () => {
  startAppListening({
    actionCreator: invocationComplete,
    effect: async (action, { dispatch, getState, take }) => {
      moduleLog.info(
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
        const { image_name, image_type } = result.image;

        // Get its URLS
        // TODO: is this extraneous? I think so...
        dispatch(
          imageUrlsReceived({ imageName: image_name, imageType: image_type })
        );

        // Get its metadata
        dispatch(
          imageMetadataReceived({
            imageName: image_name,
            imageType: image_type,
          })
        );

        // Handle canvas image
        if (
          graph_execution_state_id ===
          getState().canvas.layerState.stagingArea.sessionId
        ) {
          const [{ payload: image }] = await take(
            (
              action
            ): action is ReturnType<typeof imageMetadataReceived.fulfilled> =>
              imageMetadataReceived.fulfilled.match(action) &&
              action.payload.image_name === image_name
          );
          dispatch(addImageToStagingArea(image));
        }
      }
    },
  });
};

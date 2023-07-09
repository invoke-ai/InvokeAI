import { log } from 'app/logging/useLogger';
import { addImageToStagingArea } from 'features/canvas/store/canvasSlice';
import { progressImageSet } from 'features/system/store/systemSlice';
import { boardImagesApi } from 'services/api/endpoints/boardImages';
import { isImageOutput } from 'services/api/guards';
import { imageMetadataReceived } from 'services/api/thunks/image';
import { sessionCanceled } from 'services/api/thunks/session';
import {
  appSocketInvocationComplete,
  socketInvocationComplete,
} from 'services/events/actions';
import { startAppListening } from '../..';

const moduleLog = log.child({ namespace: 'socketio' });
const nodeDenylist = ['dataURL_image'];

export const addInvocationCompleteEventListener = () => {
  startAppListening({
    actionCreator: socketInvocationComplete,
    effect: async (action, { dispatch, getState, take }) => {
      moduleLog.debug(
        { data: action.payload },
        `Invocation complete (${action.payload.data.node.type})`
      );

      const session_id = action.payload.data.graph_execution_state_id;

      const { cancelType, isCancelScheduled, boardIdToAddTo } =
        getState().system;

      // Handle scheduled cancelation
      if (cancelType === 'scheduled' && isCancelScheduled) {
        dispatch(sessionCanceled({ session_id }));
      }

      const { data } = action.payload;
      const { result, node, graph_execution_state_id } = data;

      // This complete event has an associated image output
      if (isImageOutput(result) && !nodeDenylist.includes(node.type)) {
        const { image_name } = result.image;

        // Get its metadata
        dispatch(
          imageMetadataReceived({
            image_name,
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

        if (boardIdToAddTo && !imageDTO.is_intermediate) {
          dispatch(
            boardImagesApi.endpoints.addBoardImage.initiate({
              board_id: boardIdToAddTo,
              image_name,
            })
          );
        }

        dispatch(progressImageSet(null));
      }
      // pass along the socket event as an application action
      dispatch(appSocketInvocationComplete(action.payload));
    },
  });
};

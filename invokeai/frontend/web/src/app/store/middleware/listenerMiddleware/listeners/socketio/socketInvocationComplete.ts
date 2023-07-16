import { log } from 'app/logging/useLogger';
import { addImageToStagingArea } from 'features/canvas/store/canvasSlice';
import { IMAGE_CATEGORIES } from 'features/gallery/store/gallerySlice';
import { progressImageSet } from 'features/system/store/systemSlice';
import { boardImagesApi } from 'services/api/endpoints/boardImages';
import { imagesAdapter, imagesApi } from 'services/api/endpoints/images';
import { isImageOutput } from 'services/api/guards';
import { imageDTOReceived } from 'services/api/thunks/image';
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
          imageDTOReceived({
            image_name,
          })
        );

        const [{ payload: imageDTO }] = await take(
          imageDTOReceived.fulfilled.match
        );

        // Handle canvas image
        if (
          graph_execution_state_id ===
          getState().canvas.layerState.stagingArea.sessionId
        ) {
          dispatch(addImageToStagingArea(imageDTO));
        }

        if (!imageDTO.is_intermediate) {
          // add image to the board
          if (
            boardIdToAddTo &&
            !['all', 'none', 'batch'].includes(boardIdToAddTo)
          ) {
            dispatch(
              boardImagesApi.endpoints.addImageToBoard.initiate({
                board_id: boardIdToAddTo,
                image_name,
              })
            );
          }

          // update the cache
          const queryArg = {
            categories: IMAGE_CATEGORIES,
          };

          dispatch(
            imagesApi.util.updateQueryData('listImages', queryArg, (draft) => {
              imagesAdapter.addOne(draft, imageDTO);
              draft.total = draft.total + 1;
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

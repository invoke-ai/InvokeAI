import { log } from 'app/logging/useLogger';
import { addImageToStagingArea } from 'features/canvas/store/canvasSlice';
import {
  IMAGE_CATEGORIES,
  boardIdSelected,
  imageSelected,
} from 'features/gallery/store/gallerySlice';
import { progressImageSet } from 'features/system/store/systemSlice';
import { imagesAdapter, imagesApi } from 'services/api/endpoints/images';
import { isImageOutput } from 'services/api/guards';
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

      const { cancelType, isCancelScheduled } = getState().system;

      // Handle scheduled cancelation
      if (cancelType === 'scheduled' && isCancelScheduled) {
        dispatch(sessionCanceled({ session_id }));
      }

      const { data } = action.payload;
      const { result, node, graph_execution_state_id } = data;

      // This complete event has an associated image output
      if (isImageOutput(result) && !nodeDenylist.includes(node.type)) {
        const { image_name } = result.image;
        const { canvas, gallery } = getState();

        const imageDTO = await dispatch(
          imagesApi.endpoints.getImageDTO.initiate(image_name)
        ).unwrap();

        // Add canvas images to the staging area
        if (
          graph_execution_state_id === canvas.layerState.stagingArea.sessionId
        ) {
          dispatch(addImageToStagingArea(imageDTO));
        }

        if (!imageDTO.is_intermediate) {
          // update the cache for 'All Images'
          dispatch(
            imagesApi.util.updateQueryData(
              'listImages',
              {
                categories: IMAGE_CATEGORIES,
              },
              (draft) => {
                imagesAdapter.addOne(draft, imageDTO);
                draft.total = draft.total + 1;
              }
            )
          );

          // update the cache for 'No Board'
          dispatch(
            imagesApi.util.updateQueryData(
              'listImages',
              {
                board_id: 'none',
              },
              (draft) => {
                imagesAdapter.addOne(draft, imageDTO);
                draft.total = draft.total + 1;
              }
            )
          );

          const { autoAddBoardId } = gallery;

          // add image to the board if auto-add is enabled
          if (autoAddBoardId) {
            dispatch(
              imagesApi.endpoints.addImageToBoard.initiate({
                board_id: autoAddBoardId,
                imageDTO,
              })
            );
          }

          const { selectedBoardId, shouldAutoSwitch } = gallery;

          // If auto-switch is enabled, select the new image
          if (shouldAutoSwitch) {
            // if auto-add is enabled, switch the board as the image comes in
            if (autoAddBoardId && autoAddBoardId !== selectedBoardId) {
              dispatch(boardIdSelected(autoAddBoardId));
            } else if (!autoAddBoardId) {
              dispatch(boardIdSelected('images'));
            }
            dispatch(imageSelected(imageDTO.image_name));
          }
        }

        dispatch(progressImageSet(null));
      }
      // pass along the socket event as an application action
      dispatch(appSocketInvocationComplete(action.payload));
    },
  });
};

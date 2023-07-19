import { log } from 'app/logging/useLogger';
import { addImageToStagingArea } from 'features/canvas/store/canvasSlice';
import {
  IMAGE_CATEGORIES,
  boardIdSelected,
  imageSelected,
} from 'features/gallery/store/gallerySlice';
import { progressImageSet } from 'features/system/store/systemSlice';
import {
  SYSTEM_BOARDS,
  imagesAdapter,
  imagesApi,
} from 'services/api/endpoints/images';
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

          // add image to the board if we had one selected
          if (boardIdToAddTo && !SYSTEM_BOARDS.includes(boardIdToAddTo)) {
            dispatch(
              imagesApi.endpoints.addImageToBoard.initiate({
                board_id: boardIdToAddTo,
                imageDTO,
              })
            );
          }

          const { selectedBoardId } = gallery;

          if (boardIdToAddTo && boardIdToAddTo !== selectedBoardId) {
            dispatch(boardIdSelected(boardIdToAddTo));
          } else if (!boardIdToAddTo) {
            dispatch(boardIdSelected('all'));
          }

          // If auto-switch is enabled, select the new image
          if (getState().gallery.shouldAutoSwitch) {
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

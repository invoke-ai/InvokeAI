import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { addImageToStagingArea } from 'features/canvas/store/canvasSlice';
import {
  boardIdSelected,
  galleryViewChanged,
  imageSelected,
} from 'features/gallery/store/gallerySlice';
import { IMAGE_CATEGORIES } from 'features/gallery/store/types';
import { progressImageSet } from 'features/system/store/systemSlice';
import { imagesAdapter, imagesApi } from 'services/api/endpoints/images';
import { isImageOutput } from 'services/api/guards';
import { sessionCanceled } from 'services/api/thunks/session';
import {
  appSocketInvocationComplete,
  socketInvocationComplete,
} from 'services/events/actions';
import { startAppListening } from '../..';

const nodeDenylist = ['dataURL_image'];

export const addInvocationCompleteEventListener = () => {
  startAppListening({
    actionCreator: socketInvocationComplete,
    effect: async (action, { dispatch, getState }) => {
      const log = logger('socketio');
      const { data } = action.payload;
      log.debug(
        { data: parseify(data) },
        `Invocation complete (${action.payload.data.node.type})`
      );
      const session_id = action.payload.data.graph_execution_state_id;

      const { cancelType, isCancelScheduled } = getState().system;

      // Handle scheduled cancelation
      if (cancelType === 'scheduled' && isCancelScheduled) {
        dispatch(sessionCanceled({ session_id }));
      }

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
          /**
           * Cache updates for when an image result is received
           * - *add* to getImageDTO
           * - IF `autoAddBoardId` is set:
           *    - THEN add it to the board_id/images
           * - ELSE (`autoAddBoardId` is not set):
           *    - THEN add it to the no_board/images
           */

          const { autoAddBoardId } = gallery;
          if (autoAddBoardId) {
            dispatch(
              imagesApi.endpoints.addImageToBoard.initiate({
                board_id: autoAddBoardId,
                imageDTO,
              })
            );
          } else {
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                {
                  board_id: 'none',
                  categories: IMAGE_CATEGORIES,
                },
                (draft) => {
                  const oldTotal = draft.total;
                  const newState = imagesAdapter.addOne(draft, imageDTO);
                  const delta = newState.total - oldTotal;
                  draft.total = draft.total + delta;
                }
              )
            );
          }

          dispatch(
            imagesApi.util.invalidateTags([
              { type: 'BoardImagesTotal', id: autoAddBoardId ?? 'none' },
              { type: 'BoardAssetsTotal', id: autoAddBoardId ?? 'none' },
            ])
          );

          const { selectedBoardId, shouldAutoSwitch } = gallery;

          // If auto-switch is enabled, select the new image
          if (shouldAutoSwitch) {
            // if auto-add is enabled, switch the board as the image comes in
            if (autoAddBoardId && autoAddBoardId !== selectedBoardId) {
              dispatch(boardIdSelected(autoAddBoardId));
              dispatch(galleryViewChanged('images'));
            } else if (!autoAddBoardId) {
              dispatch(galleryViewChanged('images'));
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

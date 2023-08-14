import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { addImageToStagingArea } from 'features/canvas/store/canvasSlice';
import {
  boardIdSelected,
  galleryViewChanged,
  imageSelected,
} from 'features/gallery/store/gallerySlice';
import { IMAGE_CATEGORIES } from 'features/gallery/store/types';
import { CANVAS_OUTPUT } from 'features/nodes/util/graphBuilders/constants';
import { progressImageSet } from 'features/system/store/systemSlice';
import { imagesApi } from 'services/api/endpoints/images';
import { isImageOutput } from 'services/api/guards';
import { sessionCanceled } from 'services/api/thunks/session';
import { imagesAdapter } from 'services/api/util';
import {
  appSocketInvocationComplete,
  socketInvocationComplete,
} from 'services/events/actions';
import { startAppListening } from '../..';

const nodeDenylist = ['load_image'];

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
          graph_execution_state_id ===
            canvas.layerState.stagingArea.sessionId &&
          [CANVAS_OUTPUT].includes(data.source_node_id)
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
          if (autoAddBoardId && autoAddBoardId !== 'none') {
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
                  imagesAdapter.addOne(draft, imageDTO);
                }
              )
            );
          }

          dispatch(
            imagesApi.util.invalidateTags([
              { type: 'BoardImagesTotal', id: autoAddBoardId },
              { type: 'BoardAssetsTotal', id: autoAddBoardId },
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
            dispatch(imageSelected(imageDTO));
          }
        }

        dispatch(progressImageSet(null));
      }
      // pass along the socket event as an application action
      dispatch(appSocketInvocationComplete(action.payload));
    },
  });
};

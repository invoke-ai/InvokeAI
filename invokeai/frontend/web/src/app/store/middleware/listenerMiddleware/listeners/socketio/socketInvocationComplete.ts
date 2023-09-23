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
import { imagesApi } from 'services/api/endpoints/images';
import { isImageOutput } from 'services/api/guards';
import { imagesAdapter } from 'services/api/util';
import {
  appSocketInvocationComplete,
  socketInvocationComplete,
} from 'services/events/actions';
import { startAppListening } from '../..';

// These nodes output an image, but do not actually *save* an image, so we don't want to handle the gallery logic on them
const nodeDenylist = ['load_image', 'image'];

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

      const { result, node, queue_batch_id } = data;

      // This complete event has an associated image output
      if (isImageOutput(result) && !nodeDenylist.includes(node.type)) {
        const { image_name } = result.image;
        const { canvas, gallery } = getState();

        // This populates the `getImageDTO` cache
        const imageDTO = await dispatch(
          imagesApi.endpoints.getImageDTO.initiate(image_name)
        ).unwrap();

        // Add canvas images to the staging area
        if (
          canvas.batchIds.includes(queue_batch_id) &&
          [CANVAS_OUTPUT].includes(data.source_node_id)
        ) {
          dispatch(addImageToStagingArea(imageDTO));
        }

        if (!imageDTO.is_intermediate) {
          /**
           * Cache updates for when an image result is received
           * - add it to the no_board/images
           */

          dispatch(
            imagesApi.util.updateQueryData(
              'listImages',
              {
                board_id: imageDTO.board_id ?? 'none',
                categories: IMAGE_CATEGORIES,
              },
              (draft) => {
                imagesAdapter.addOne(draft, imageDTO);
              }
            )
          );

          dispatch(
            imagesApi.util.invalidateTags([
              { type: 'BoardImagesTotal', id: imageDTO.board_id },
              { type: 'BoardAssetsTotal', id: imageDTO.board_id },
            ])
          );

          const { shouldAutoSwitch } = gallery;

          // If auto-switch is enabled, select the new image
          if (shouldAutoSwitch) {
            // if auto-add is enabled, switch the board as the image comes in
            dispatch(galleryViewChanged('images'));
            dispatch(boardIdSelected(imageDTO.board_id ?? 'none'));
            dispatch(imageSelected(imageDTO));
          }
        }
      }
      // pass along the socket event as an application action
      dispatch(appSocketInvocationComplete(action.payload));
    },
  });
};

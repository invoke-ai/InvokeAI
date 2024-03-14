import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { deepClone } from 'common/util/deepClone';
import { parseify } from 'common/util/serialize';
import { addImageToStagingArea } from 'features/canvas/store/canvasSlice';
import {
  boardIdSelected,
  galleryViewChanged,
  imageSelected,
  isImageViewerOpenChanged,
} from 'features/gallery/store/gallerySlice';
import { IMAGE_CATEGORIES } from 'features/gallery/store/types';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useExecutionState';
import { isImageOutput } from 'features/nodes/types/common';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { CANVAS_OUTPUT } from 'features/nodes/util/graph/constants';
import { boardsApi } from 'services/api/endpoints/boards';
import { imagesApi } from 'services/api/endpoints/images';
import { imagesAdapter } from 'services/api/util';
import { socketInvocationComplete } from 'services/events/actions';

// These nodes output an image, but do not actually *save* an image, so we don't want to handle the gallery logic on them
const nodeTypeDenylist = ['load_image', 'image'];

const log = logger('socketio');

export const addInvocationCompleteEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketInvocationComplete,
    effect: async (action, { dispatch, getState }) => {
      const { data } = action.payload;
      log.debug({ data: parseify(data) }, `Invocation complete (${data.invocation_type})`);

      const { result, invocation_source_id } = data;
      // This complete event has an associated image output
      if (isImageOutput(data.result) && !nodeTypeDenylist.includes(data.invocation_type)) {
        const { image_name } = data.result.image;
        const { canvas, gallery } = getState();

        // This populates the `getImageDTO` cache
        const imageDTORequest = dispatch(
          imagesApi.endpoints.getImageDTO.initiate(image_name, {
            forceRefetch: true,
          })
        );

        const imageDTO = await imageDTORequest.unwrap();
        imageDTORequest.unsubscribe();

        // Add canvas images to the staging area
        if (canvas.batchIds.includes(data.batch_id) && data.invocation_source_id === CANVAS_OUTPUT) {
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

          // update the total images for the board
          dispatch(
            boardsApi.util.updateQueryData('getBoardImagesTotal', imageDTO.board_id ?? 'none', (draft) => {
              // eslint-disable-next-line @typescript-eslint/no-unused-vars
              draft.total += 1;
            })
          );

          dispatch(imagesApi.util.invalidateTags([{ type: 'Board', id: imageDTO.board_id ?? 'none' }]));

          const { shouldAutoSwitch } = gallery;

          // If auto-switch is enabled, select the new image
          if (shouldAutoSwitch) {
            // if auto-add is enabled, switch the gallery view and board if needed as the image comes in
            if (gallery.galleryView !== 'images') {
              dispatch(galleryViewChanged('images'));
            }

            if (imageDTO.board_id && imageDTO.board_id !== gallery.selectedBoardId) {
              dispatch(
                boardIdSelected({
                  boardId: imageDTO.board_id,
                  selectedImageName: imageDTO.image_name,
                })
              );
            }

            if (!imageDTO.board_id && gallery.selectedBoardId !== 'none') {
              dispatch(
                boardIdSelected({
                  boardId: 'none',
                  selectedImageName: imageDTO.image_name,
                })
              );
            }

            dispatch(imageSelected(imageDTO));
            dispatch(isImageViewerOpenChanged(true));
          }
        }
      }

      const nes = deepClone($nodeExecutionStates.get()[invocation_source_id]);
      if (nes) {
        nes.status = zNodeStatus.enum.COMPLETED;
        if (nes.progress !== null) {
          nes.progress = 1;
        }
        nes.outputs.push(result);
        upsertExecutionState(nes.nodeId, nes);
      }
    },
  });
};

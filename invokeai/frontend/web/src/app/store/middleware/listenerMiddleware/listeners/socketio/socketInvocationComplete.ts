import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { deepClone } from 'common/util/deepClone';
import { parseify } from 'common/util/serialize';
import { stagingAreaImageAdded } from 'features/controlLayers/store/canvasV2Slice';
import { boardIdSelected, galleryViewChanged, imageSelected, offsetChanged } from 'features/gallery/store/gallerySlice';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useExecutionState';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { CANVAS_OUTPUT } from 'features/nodes/util/graph/constants';
import { boardsApi } from 'services/api/endpoints/boards';
import { imagesApi } from 'services/api/endpoints/images';
import { getCategories, getListImagesUrl } from 'services/api/util';
import { socketInvocationComplete } from 'services/events/actions';

// These nodes output an image, but do not actually *save* an image, so we don't want to handle the gallery logic on them
const nodeTypeDenylist = ['load_image', 'image'];

const log = logger('socketio');

export const addInvocationCompleteEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketInvocationComplete,
    effect: async (action, { dispatch, getState }) => {
      const { data } = action.payload;
      log.debug({ data: parseify(data) }, `Invocation complete (${data.invocation.type})`);

      const { result, invocation_source_id } = data;
      // This complete event has an associated image output
      if (data.result.type === 'image_output' && !nodeTypeDenylist.includes(data.invocation.type)) {
        const { image_name } = data.result.image;
        const { gallery, canvasV2 } = getState();

        // This populates the `getImageDTO` cache
        const imageDTORequest = dispatch(
          imagesApi.endpoints.getImageDTO.initiate(image_name, {
            forceRefetch: true,
          })
        );

        const imageDTO = await imageDTORequest.unwrap();
        imageDTORequest.unsubscribe();

        // handle tab-specific logic
        if (data.origin === 'canvas') {
          if (data.invocation_source_id === CANVAS_OUTPUT && canvasV2.stagingArea.isStaging) {
            dispatch(stagingAreaImageAdded({ imageDTO }));
          }
        } else if (data.origin === 'workflows') {
          const nes = deepClone($nodeExecutionStates.get()[invocation_source_id]);
          if (nes) {
            nes.status = zNodeStatus.enum.COMPLETED;
            if (nes.progress !== null) {
              nes.progress = 1;
            }
            nes.outputs.push(result);
            upsertExecutionState(nes.nodeId, nes);
          }
        }

        if (!imageDTO.is_intermediate) {
          // update the total images for the board
          dispatch(
            boardsApi.util.updateQueryData('getBoardImagesTotal', imageDTO.board_id ?? 'none', (draft) => {
              // eslint-disable-next-line @typescript-eslint/no-unused-vars
              draft.total += 1;
            })
          );

          dispatch(
            imagesApi.util.invalidateTags([
              { type: 'Board', id: imageDTO.board_id ?? 'none' },
              {
                type: 'ImageList',
                id: getListImagesUrl({
                  board_id: imageDTO.board_id ?? 'none',
                  categories: getCategories(imageDTO),
                }),
              },
            ])
          );

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

            dispatch(offsetChanged({ offset: 0 }));

            if (!imageDTO.board_id && gallery.selectedBoardId !== 'none') {
              dispatch(
                boardIdSelected({
                  boardId: 'none',
                  selectedImageName: imageDTO.image_name,
                })
              );
            }

            dispatch(imageSelected(imageDTO));
          }
        }
      }
    },
  });
};

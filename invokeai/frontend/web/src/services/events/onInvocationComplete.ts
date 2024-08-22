import { logger } from 'app/logging/logger';
import type { AppDispatch, RootState } from 'app/store/store';
import type { SerializableObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import { sessionImageStaged } from 'features/controlLayers/store/canvasV2Slice';
import { boardIdSelected, galleryViewChanged, imageSelected, offsetChanged } from 'features/gallery/store/gallerySlice';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useExecutionState';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { boardsApi } from 'services/api/endpoints/boards';
import { getImageDTO, imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { getCategories, getListImagesUrl } from 'services/api/util';
import type { InvocationCompleteEvent, InvocationDenoiseProgressEvent } from 'services/events/types';

const log = logger('events');

export const buildOnInvocationComplete = (
  getState: () => RootState,
  dispatch: AppDispatch,
  nodeTypeDenylist: string[],
  setLastProgressEvent: (event: InvocationDenoiseProgressEvent | null) => void,
  setLastCanvasProgressEvent: (event: InvocationDenoiseProgressEvent | null) => void
) => {
  const addImageToGallery = (imageDTO: ImageDTO) => {
    if (imageDTO.is_intermediate) {
      return;
    }

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

    const { shouldAutoSwitch, galleryView, selectedBoardId } = getState().gallery;

    // If auto-switch is enabled, select the new image
    if (shouldAutoSwitch) {
      // if auto-add is enabled, switch the gallery view and board if needed as the image comes in
      if (galleryView !== 'images') {
        dispatch(galleryViewChanged('images'));
      }

      if (imageDTO.board_id && imageDTO.board_id !== selectedBoardId) {
        dispatch(
          boardIdSelected({
            boardId: imageDTO.board_id,
            selectedImageName: imageDTO.image_name,
          })
        );
      }

      dispatch(offsetChanged({ offset: 0 }));

      if (!imageDTO.board_id && selectedBoardId !== 'none') {
        dispatch(
          boardIdSelected({
            boardId: 'none',
            selectedImageName: imageDTO.image_name,
          })
        );
      }

      dispatch(imageSelected(imageDTO));
    }
  };

  return async (data: InvocationCompleteEvent) => {
    log.debug(
      { data } as SerializableObject,
      `Invocation complete (${data.invocation.type}, ${data.invocation_source_id})`
    );

    const { result, invocation_source_id } = data;

    // Update the node execution states - the image output is handled below
    if (data.origin === 'workflows') {
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

    // This complete event has an associated image output
    if (
      (data.result.type === 'image_output' || data.result.type === 'canvas_v2_mask_and_crop_output') &&
      !nodeTypeDenylist.includes(data.invocation.type)
    ) {
      const { image_name } = data.result.image;
      const { session } = getState().canvasV2;

      const imageDTO = await getImageDTO(image_name);

      if (!imageDTO) {
        log.error({ data } as SerializableObject, 'Failed to fetch image DTO after generation');
        return;
      }

      if (data.origin === 'canvas') {
        if (data.invocation_source_id !== 'canvas_output') {
          // Not a canvas output image - ignore
          return;
        }
        if (session.mode === 'compose' && session.isStaging) {
          if (data.result.type === 'canvas_v2_mask_and_crop_output') {
            const { offset_x, offset_y } = data.result;
            if (session.isStaging) {
              dispatch(sessionImageStaged({ stagingAreaImage: { imageDTO, offsetX: offset_x, offsetY: offset_y } }));
            }
          } else if (data.result.type === 'image_output') {
            if (session.isStaging) {
              dispatch(sessionImageStaged({ stagingAreaImage: { imageDTO, offsetX: 0, offsetY: 0 } }));
            }
          }
          addImageToGallery(imageDTO);
        } else {
          addImageToGallery(imageDTO);
          setLastCanvasProgressEvent(null);
        }
      }
    }

    setLastProgressEvent(null);
  };
};

import { logger } from 'app/logging/logger';
import type { AppDispatch, RootState } from 'app/store/store';
import type { SerializableObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import { stagingAreaImageStaged } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { boardIdSelected, galleryViewChanged, imageSelected, offsetChanged } from 'features/gallery/store/gallerySlice';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useExecutionState';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { boardsApi } from 'services/api/endpoints/boards';
import { getImageDTO, imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO, S } from 'services/api/types';
import { getCategories, getListImagesUrl } from 'services/api/util';

const log = logger('events');

const isCanvasOutputNode = (data: S['InvocationCompleteEvent']) => {
  return data.invocation_source_id.split(':')[0] === 'canvas_output';
};

const nodeTypeDenylist = ['load_image', 'image'];

export const buildOnInvocationComplete = (getState: () => RootState, dispatch: AppDispatch) => {
  const addImageToGallery = (data: S['InvocationCompleteEvent'], imageDTO: ImageDTO) => {
    if (nodeTypeDenylist.includes(data.invocation.type)) {
      log.trace('Skipping node type denylisted');
      return;
    }

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

  const getResultImageDTO = (data: S['InvocationCompleteEvent']) => {
    const { result } = data;
    if (result.type === 'image_output') {
      return getImageDTO(result.image.image_name);
    } else if (result.type === 'canvas_v2_mask_and_crop_output') {
      return getImageDTO(result.image.image_name);
    }
    return null;
  };

  const handleOriginWorkflows = async (data: S['InvocationCompleteEvent']) => {
    const { result, invocation_source_id } = data;

    const nes = deepClone($nodeExecutionStates.get()[invocation_source_id]);
    if (nes) {
      nes.status = zNodeStatus.enum.COMPLETED;
      if (nes.progress !== null) {
        nes.progress = 1;
      }
      nes.outputs.push(result);
      upsertExecutionState(nes.nodeId, nes);
    }

    const imageDTO = await getResultImageDTO(data);

    if (imageDTO && !imageDTO.is_intermediate) {
      addImageToGallery(data, imageDTO);
    }
  };

  const handleOriginCanvas = async (data: S['InvocationCompleteEvent']) => {
    const imageDTO = await getResultImageDTO(data);

    if (!imageDTO) {
      return;
    }

    if (data.destination === 'canvas') {
      // TODO(psyche): Can/should we let canvas handle this itself?
      if (isCanvasOutputNode(data)) {
        if (data.result.type === 'canvas_v2_mask_and_crop_output') {
          const { offset_x, offset_y } = data.result;
          dispatch(stagingAreaImageStaged({ stagingAreaImage: { imageDTO, offsetX: offset_x, offsetY: offset_y } }));
        } else if (data.result.type === 'image_output') {
          dispatch(stagingAreaImageStaged({ stagingAreaImage: { imageDTO, offsetX: 0, offsetY: 0 } }));
        }
        addImageToGallery(data, imageDTO);
      }
    } else if (!imageDTO.is_intermediate) {
      // Desintaion is gallery
      addImageToGallery(data, imageDTO);
    }
  };

  const handleOriginOther = async (data: S['InvocationCompleteEvent']) => {
    const imageDTO = await getResultImageDTO(data);

    if (imageDTO && !imageDTO.is_intermediate) {
      addImageToGallery(data, imageDTO);
    }
  };

  return async (data: S['InvocationCompleteEvent']) => {
    log.debug(
      { data } as SerializableObject,
      `Invocation complete (${data.invocation.type}, ${data.invocation_source_id})`
    );

    if (data.origin === 'workflows') {
      await handleOriginWorkflows(data);
    } else if (data.origin === 'canvas') {
      await handleOriginCanvas(data);
    } else {
      await handleOriginOther(data);
    }
  };
};

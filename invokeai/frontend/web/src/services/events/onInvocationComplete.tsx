import { logger } from 'app/logging/logger';
import type { AppDispatch, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { stagingAreaImageStaged } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { boardIdSelected, galleryViewChanged, imageSelected, offsetChanged } from 'features/gallery/store/gallerySlice';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useNodeExecutionState';
import { isImageField, isImageFieldCollection } from 'features/nodes/types/common';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { isCanvasOutputEvent } from 'features/nodes/util/graph/graphBuilderUtils';
import { flushSync } from 'react-dom';
import type { ApiTagDescription } from 'services/api';
import { boardsApi } from 'services/api/endpoints/boards';
import { getImageDTOSafe, imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO, S } from 'services/api/types';
import { getCategories, getListImagesUrl } from 'services/api/util';
import { $lastCanvasProgressImage, $lastProgressEvent } from 'services/events/stores';
import type { Param0 } from 'tsafe';
import { objectEntries } from 'tsafe';
import type { JsonObject } from 'type-fest';

const log = logger('events');

const nodeTypeDenylist = ['load_image', 'image'];

export const buildOnInvocationComplete = (getState: () => RootState, dispatch: AppDispatch) => {
  const addImagesToGallery = (data: S['InvocationCompleteEvent'], imageDTOs: ImageDTO[]) => {
    if (nodeTypeDenylist.includes(data.invocation.type)) {
      log.trace(`Skipping denylisted node type (${data.invocation.type})`);
      return;
    }

    // For efficiency's sake, we want to minimize the number of dispatches and invalidations we do.
    // We'll keep track of each change we need to make and do them all at once.
    const boardTotalAdditions: Record<string, number> = {};
    const boardTagIdsToInvalidate: Set<string> = new Set();
    const imageListTagIdsToInvalidate: Set<string> = new Set();

    for (const imageDTO of imageDTOs) {
      if (imageDTO.is_intermediate) {
        return;
      }

      const boardId = imageDTO.board_id ?? 'none';
      // update the total images for the board
      boardTotalAdditions[boardId] = (boardTotalAdditions[boardId] || 0) + 1;
      // invalidate the board tag
      boardTagIdsToInvalidate.add(boardId);
      // invalidate the image list tag
      imageListTagIdsToInvalidate.add(
        getListImagesUrl({
          board_id: boardId,
          categories: getCategories(imageDTO),
        })
      );
    }

    // Update all the board image totals at once
    const entries: Param0<typeof boardsApi.util.upsertQueryEntries> = [];
    for (const [boardId, amountToAdd] of objectEntries(boardTotalAdditions)) {
      // upsertQueryEntries doesn't provide a "recipe" function for the update - we must provide the new value
      // directly. So we need to select the board totals first.
      const total = boardsApi.endpoints.getBoardImagesTotal.select(boardId)(getState()).data?.total;
      if (total === undefined) {
        // No cache exists for this board, so we can't update it.
        continue;
      }
      entries.push({
        endpointName: 'getBoardImagesTotal',
        arg: boardId,
        value: { total: total + amountToAdd },
      });
    }
    dispatch(boardsApi.util.upsertQueryEntries(entries));

    // Invalidate all tags at once
    const boardTags: ApiTagDescription[] = Array.from(boardTagIdsToInvalidate).map((boardId) => ({
      type: 'Board' as const,
      id: boardId,
    }));
    const imageListTags: ApiTagDescription[] = Array.from(imageListTagIdsToInvalidate).map((imageListId) => ({
      type: 'ImageList' as const,
      id: imageListId,
    }));
    dispatch(imagesApi.util.invalidateTags([...boardTags, ...imageListTags]));

    // Finally, we may need to autoswitch to the new image. We'll only do it for the last image in the list.

    const lastImageDTO = imageDTOs.at(-1);

    if (!lastImageDTO) {
      return;
    }

    const { image_name, board_id } = lastImageDTO;

    const { shouldAutoSwitch, selectedBoardId, galleryView, offset } = getState().gallery;

    // If auto-switch is enabled, select the new image
    if (shouldAutoSwitch) {
      // If the image is from a different board, switch to that board - this will also select the image
      if (board_id && board_id !== selectedBoardId) {
        dispatch(
          boardIdSelected({
            boardId: board_id,
            selectedImageName: image_name,
          })
        );
      } else if (!board_id && selectedBoardId !== 'none') {
        dispatch(
          boardIdSelected({
            boardId: 'none',
            selectedImageName: image_name,
          })
        );
      } else {
        // Else just select the image, no need to switch boards
        dispatch(imageSelected(lastImageDTO));

        if (galleryView !== 'images') {
          // We also need to update the gallery view to images. This also updates the offset.
          dispatch(galleryViewChanged('images'));
        } else if (offset > 0) {
          // If we are not at the start of the gallery, reset the offset.
          dispatch(offsetChanged({ offset: 0 }));
        }
      }
    }
  };

  const getResultImageDTOs = async (data: S['InvocationCompleteEvent']): Promise<ImageDTO[]> => {
    const { result } = data;
    const imageDTOs: ImageDTO[] = [];
    for (const [_name, value] of objectEntries(result)) {
      if (isImageField(value)) {
        const imageDTO = await getImageDTOSafe(value.image_name);
        if (imageDTO) {
          imageDTOs.push(imageDTO);
        }
      } else if (isImageFieldCollection(value)) {
        for (const imageField of value) {
          const imageDTO = await getImageDTOSafe(imageField.image_name);
          if (imageDTO) {
            imageDTOs.push(imageDTO);
          }
        }
      }
    }
    return imageDTOs;
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

    const imageDTOs = await getResultImageDTOs(data);
    addImagesToGallery(data, imageDTOs);
  };

  const handleOriginCanvas = async (data: S['InvocationCompleteEvent']) => {
    if (!isCanvasOutputEvent(data)) {
      return;
    }

    // We expect only a single image in the canvas output
    const imageDTO = (await getResultImageDTOs(data))[0];

    if (!imageDTO) {
      return;
    }

    flushSync(() => {
      dispatch(stagingAreaImageStaged({ stagingAreaImage: { imageDTO, offsetX: 0, offsetY: 0 } }));
    });

    $lastCanvasProgressImage.set(null);
  };

  const handleOriginOther = async (data: S['InvocationCompleteEvent']) => {
    const imageDTOs = await getResultImageDTOs(data);
    addImagesToGallery(data, imageDTOs);
  };

  return async (data: S['InvocationCompleteEvent']) => {
    log.debug({ data } as JsonObject, `Invocation complete (${data.invocation.type}, ${data.invocation_source_id})`);

    if (data.origin === 'workflows') {
      await handleOriginWorkflows(data);
    } else if (data.origin === 'canvas') {
      await handleOriginCanvas(data);
    } else {
      await handleOriginOther(data);
    }

    $lastProgressEvent.set(null);
  };
};

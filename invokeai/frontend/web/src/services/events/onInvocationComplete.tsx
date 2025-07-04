import { logger } from 'app/logging/logger';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import {
  selectAutoSwitch,
  selectGalleryView,
  selectListImageNamesQueryArgs,
  selectSelectedBoardId,
} from 'features/gallery/store/gallerySelectors';
import { boardIdSelected, galleryViewChanged, imageSelected } from 'features/gallery/store/gallerySlice';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useNodeExecutionState';
import { isImageField, isImageFieldCollection } from 'features/nodes/types/common';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { boardsApi } from 'services/api/endpoints/boards';
import { getImageDTOSafe, imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO, S } from 'services/api/types';
import { getCategories } from 'services/api/util';
import { insertImageIntoNamesResult } from 'services/api/util/optimisticUpdates';
import { $lastProgressEvent } from 'services/events/stores';
import type { Param0 } from 'tsafe';
import { objectEntries } from 'tsafe';
import type { JsonObject } from 'type-fest';

const log = logger('events');

const nodeTypeDenylist = ['load_image', 'image'];

export const buildOnInvocationComplete = (getState: AppGetState, dispatch: AppDispatch) => {
  const addImagesToGallery = async (data: S['InvocationCompleteEvent']) => {
    if (nodeTypeDenylist.includes(data.invocation.type)) {
      log.trace(`Skipping denylisted node type (${data.invocation.type})`);
      return;
    }

    const imageDTOs = await getResultImageDTOs(data);
    if (imageDTOs.length === 0) {
      return;
    }

    // For efficiency's sake, we want to minimize the number of dispatches and invalidations we do.
    // We'll keep track of each change we need to make and do them all at once.
    const boardTotalAdditions: Record<string, number> = {};
    const listImageNamesArg = selectListImageNamesQueryArgs(getState());

    for (const imageDTO of imageDTOs) {
      if (imageDTO.is_intermediate) {
        return;
      }

      const board_id = imageDTO.board_id ?? 'none';
      // update the total images for the board
      boardTotalAdditions[board_id] = (boardTotalAdditions[board_id] || 0) + 1;
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

    // Optimistically update image names lists - DTOs are already cached by getResultImageDTOs
    const state = getState();

    for (const imageDTO of imageDTOs) {
      // Construct the expected query args for this image's getImageNames query
      // Use the current gallery query args as base, but override board_id and categories for this specific image
      const expectedQueryArgs = {
        ...listImageNamesArg,
        // We don't have enough information to do optimistic updates when a search term is present.
        search_term: '',
        categories: getCategories(imageDTO),
        board_id: imageDTO.board_id ?? 'none',
      };

      // Check if we have cached image names for this query
      const cachedNamesResult = imagesApi.endpoints.getImageNames.select(expectedQueryArgs)(state);

      if (cachedNamesResult.data) {
        // We have cached names - optimistically insert the new image
        dispatch(
          imagesApi.util.updateQueryData('getImageNames', expectedQueryArgs, (draft) => {
            // Use the utility function to insert at the correct position
            const updatedResult = insertImageIntoNamesResult(
              draft,
              imageDTO,
              expectedQueryArgs.starred_first ?? true,
              expectedQueryArgs.order_dir
            );

            // Replace the draft contents
            draft.image_names = updatedResult.image_names;
            draft.starred_count = updatedResult.starred_count;
            draft.total_count = updatedResult.total_count;
          })
        );
      }
      // If no cached data, we don't need to do anything - there's no list to update
    }

    // No need to invalidate tags since we're doing optimistic updates
    // Board totals are already updated above via upsertQueryEntries

    const autoSwitch = selectAutoSwitch(getState());

    if (!autoSwitch) {
      return;
    }

    // Finally, we may need to autoswitch to the new image. We'll only do it for the last image in the list.
    const lastImageDTO = imageDTOs.at(-1);

    if (!lastImageDTO) {
      return;
    }

    const { image_name } = lastImageDTO;
    const board_id = lastImageDTO.board_id ?? 'none';

    // With optimistic updates, we can immediately switch to the new image
    const selectedBoardId = selectSelectedBoardId(getState());

    // If the image is from a different board, switch to that board & select the image - otherwise just select the
    // image. This implicitly changes the view to 'images' if it was not already.
    if (board_id !== selectedBoardId) {
      dispatch(
        boardIdSelected({
          boardId: board_id,
          selectedImageName: image_name,
        })
      );
    } else {
      // Ensure we are on the 'images' gallery view - that's where this image will be displayed
      const galleryView = selectGalleryView(getState());
      if (galleryView !== 'images') {
        dispatch(galleryViewChanged('images'));
      }
      // Select the image immediately since we've optimistically updated the cache
      dispatch(imageSelected(lastImageDTO.image_name));
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

  return async (data: S['InvocationCompleteEvent']) => {
    log.debug({ data } as JsonObject, `Invocation complete (${data.invocation.type}, ${data.invocation_source_id})`);

    const nodeExecutionState = $nodeExecutionStates.get()[data.invocation_source_id];

    if (nodeExecutionState) {
      const _nodeExecutionState = deepClone(nodeExecutionState);
      _nodeExecutionState.status = zNodeStatus.enum.COMPLETED;
      if (_nodeExecutionState.progress !== null) {
        _nodeExecutionState.progress = 1;
      }
      _nodeExecutionState.outputs.push(data.result);
      upsertExecutionState(_nodeExecutionState.nodeId, _nodeExecutionState);
    }

    await addImagesToGallery(data);

    $lastProgressEvent.set(null);
  };
};

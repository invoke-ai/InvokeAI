import { logger } from 'app/logging/logger';
import { addAppListener } from 'app/store/middleware/listenerMiddleware';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import {
  selectAutoSwitch,
  selectGalleryView,
  selectListImagesBaseQueryArgs,
  selectSelectedBoardId,
} from 'features/gallery/store/gallerySelectors';
import { boardIdSelected, galleryViewChanged, imageSelected } from 'features/gallery/store/gallerySlice';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useNodeExecutionState';
import { isImageField, isImageFieldCollection } from 'features/nodes/types/common';
import { zNodeStatus } from 'features/nodes/types/invocation';
import type { ApiTagDescription } from 'services/api';
import { boardsApi } from 'services/api/endpoints/boards';
import { getImageDTOSafe, imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO, S } from 'services/api/types';
import { getCategories } from 'services/api/util';
import { $lastProgressEvent } from 'services/events/stores';
import stableHash from 'stable-hash';
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
    const boardTagIdsToInvalidate: Set<string> = new Set();
    const imageListTagIdsToInvalidate: Set<string> = new Set();
    const listImagesArg = selectListImagesBaseQueryArgs(getState());

    for (const imageDTO of imageDTOs) {
      if (imageDTO.is_intermediate) {
        return;
      }

      const board_id = imageDTO.board_id ?? 'none';
      // update the total images for the board
      boardTotalAdditions[board_id] = (boardTotalAdditions[board_id] || 0) + 1;
      // invalidate the board tag
      boardTagIdsToInvalidate.add(board_id);
      // invalidate the image list tag
      imageListTagIdsToInvalidate.add(
        stableHash({
          ...listImagesArg,
          categories: getCategories(imageDTO),
          board_id,
          offset: 0,
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
    dispatch(imagesApi.util.invalidateTags(['ImageNameList', ...boardTags, ...imageListTags]));

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

    /**
     * Auto-switch needs a bit of care to avoid race conditions - we need to invalidate the appropriate image list
     * query cache, and only after it has loaded, select the new image.
     */
    const queryArgs = {
      ...listImagesArg,
      categories: getCategories(lastImageDTO),
      board_id,
      offset: 0,
    };

    dispatch(
      addAppListener({
        predicate: (action) => {
          if (!imagesApi.endpoints.listImages.matchFulfilled(action)) {
            return false;
          }

          if (stableHash(action.meta.arg.originalArgs) !== stableHash(queryArgs)) {
            return false;
          }

          return true;
        },
        effect: (_action, { getState, dispatch, unsubscribe }) => {
          // This is a one-time listener - we always unsubscribe after the first match
          unsubscribe();

          // Auto-switch may have been disabled while we were waiting for the query to resolve - bail if so
          const autoSwitch = selectAutoSwitch(getState());
          if (!autoSwitch) {
            return;
          }

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
            // Else just select the image, no need to switch boards
            dispatch(imageSelected(lastImageDTO.image_name));
          }
        },
      })
    );
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

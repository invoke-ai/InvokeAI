import { logger } from 'app/logging/logger';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import {
  selectAutoSwitch,
  selectGalleryView,
  selectGetImageNamesQueryArgs,
  selectSelectedBoardId,
} from 'features/gallery/store/gallerySelectors';
import { boardIdSelected, galleryViewChanged, imageSelected } from 'features/gallery/store/gallerySlice';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useNodeExecutionState';
import { isImageField, isImageFieldCollection } from 'features/nodes/types/common';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { generatedVideoChanged } from 'features/parameters/store/videoSlice';
import type { LRUCache } from 'lru-cache';
import { boardsApi } from 'services/api/endpoints/boards';
import { getImageDTOSafe, imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO, S } from 'services/api/types';
import { getCategories } from 'services/api/util';
import { insertImageIntoNamesResult } from 'services/api/util/optimisticUpdates';
import { $lastProgressEvent } from 'services/events/stores';
import stableHash from 'stable-hash';
import type { Param0 } from 'tsafe';
import { objectEntries } from 'tsafe';
import type { JsonObject } from 'type-fest';

const log = logger('events');

const nodeTypeDenylist = ['load_image', 'image'];

export const buildOnInvocationComplete = (
  getState: AppGetState,
  dispatch: AppDispatch,
  finishedQueueItemIds: LRUCache<number, boolean>
) => {
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
    const getImageNamesArg = selectGetImageNamesQueryArgs(getState());

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

    /**
     * Optimistic update and cache invalidation for image names queries that match this image's board and categories.
     * - Optimistic update for the cache that does not have a search term (we cannot derive the correct insertion
     *   position when a search term is present).
     * - Cache invalidation for the query that has a search term, so it will be refetched.
     *
     * Note: The image DTO objects are already implicitly cached by the getResultImageDTOs function. We do not need
     * to explicitly cache them again here.
     */
    for (const imageDTO of imageDTOs) {
      // Override board_id and categories for this specific image to build the "expected" args for the query.
      const imageSpecificArgs = {
        categories: getCategories(imageDTO),
        board_id: imageDTO.board_id ?? 'none',
      };

      const expectedQueryArgs = {
        ...getImageNamesArg,
        ...imageSpecificArgs,
        search_term: '',
      };

      // If the cache for the query args provided here does not exist, RTK Query will ignore the update.
      dispatch(
        imagesApi.util.updateQueryData(
          'getImageNames',
          {
            ...getImageNamesArg,
            ...imageSpecificArgs,
            search_term: '',
          },
          (draft) => {
            const updatedResult = insertImageIntoNamesResult(
              draft,
              imageDTO,
              expectedQueryArgs.starred_first ?? true,
              expectedQueryArgs.order_dir
            );

            draft.image_names = updatedResult.image_names;
            draft.starred_count = updatedResult.starred_count;
            draft.total_count = updatedResult.total_count;
          }
        )
      );

      // If there is a search term present, we need to invalidate that query to ensure the search results are updated.
      if (getImageNamesArg.search_term) {
        const expectedQueryArgs = {
          ...getImageNamesArg,
          ...imageSpecificArgs,
        };
        dispatch(imagesApi.util.invalidateTags([{ type: 'ImageNameList', id: stableHash(expectedQueryArgs) }]));
      }
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

  const getResultVideoDTOs = async (data: S['InvocationCompleteEvent']): Promise<{url: string , taskId: number} | null> => {
    // @ts-expect-error: This is a workaround to get the video name from the result
    if (data.invocation.type === 'runway_generate_video') {
      // @ts-expect-error: This is a workaround to get the video name from the result
      return {url: data.result.video_url, taskId: data.result.runway_task_id};
    }
    return null;
  };

  return async (data: S['InvocationCompleteEvent']) => {
    if (finishedQueueItemIds.has(data.item_id)) {
      log.trace({ data } as JsonObject, `Received event for already-finished queue item ${data.item_id}`);
      return;
    }
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

    const videoResult = await getResultVideoDTOs(data);
    if (videoResult) {
      dispatch(generatedVideoChanged(videoResult));
    }

    $lastProgressEvent.set(null);
  };
};

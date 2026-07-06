import { logger } from 'app/logging/logger';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { canvasWorkflowIntegrationProcessingCompleted } from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import {
  selectAutoSwitch,
  selectGalleryView,
  selectGetImageNamesQueryArgs,
  selectListBoardsQueryArgs,
  selectSelectedBoardId,
} from 'features/gallery/store/gallerySelectors';
import { boardIdSelected, galleryViewChanged, imageSelected } from 'features/gallery/store/gallerySlice';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useNodeExecutionState';
import { isImageField, isImageFieldCollection, isVideoField } from 'features/nodes/types/common';
import { LIST_ALL_TAG } from 'services/api';
import { boardsApi } from 'services/api/endpoints/boards';
import { galleryApi } from 'services/api/endpoints/gallery';
import { getImageDTOSafe, imagesApi } from 'services/api/endpoints/images';
import { queueApi } from 'services/api/endpoints/queue';
import { getVideoDTOSafe } from 'services/api/endpoints/videos';
import type { ImageDTO, S, VideoDTO } from 'services/api/types';
import { getCategories } from 'services/api/util';
import { insertImageIntoNamesResult } from 'services/api/util/optimisticUpdates';
import { getTagsToInvalidateForBoardAffectingMutation } from 'services/api/util/tagInvalidation';
import { getUpdatedNodeExecutionStateOnInvocationComplete } from 'services/events/nodeExecutionState';
import { $lastProgressEvent } from 'services/events/stores';
import stableHash from 'stable-hash';
import type { Param0 } from 'tsafe';
import { objectEntries } from 'tsafe';
import type { JsonObject } from 'type-fest';

const log = logger('events');

// These nodes are passthrough nodes. They do not add images to the gallery, so we must skip that handling for them.
const nodeTypeDenylist = ['load_image', 'image'];

/**
 * Builds the socket event handler for invocation complete events. Adds output images to the gallery and/or updates
 * node execution states for the workflow editor.
 *
 * @param getState The Redux getState function.
 * @param dispatch The Redux dispatch function.
 * @param completedInvocationKeysByItemId A listener-local map used to dedupe repeated invocation completion events
 * and to share completion knowledge with the other invocation event handlers.
 */
export const buildOnInvocationComplete = (
  getState: AppGetState,
  dispatch: AppDispatch,
  completedInvocationKeysByItemId: Map<number, Set<string>>
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

    dispatch(
      boardsApi.util.updateQueryData('listAllBoards', selectListBoardsQueryArgs(getState()), (draft) => {
        for (const board of draft) {
          board.image_count = board.image_count + (boardTotalAdditions[board.board_id] ?? 0);
        }
      })
    );

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
    // Exception: virtual board groupings aren't covered by the optimistic updates above, so
    // their counts/cover thumbnails would otherwise lag behind until the next mutation.
    if (Object.keys(boardTotalAdditions).length > 0) {
      dispatch(imagesApi.util.invalidateTags(['VirtualBoards']));
    }

    // The optimistic updates above only touch the image-only ``getImageNames`` cache. The
    // gallery grid actually subscribes to the polymorphic ``getGalleryItemNames`` endpoint
    // (which interleaves images and videos by created_at), so without invalidating its tag
    // a freshly generated image never appears in the grid until the user reloads — even
    // though it lands correctly in board totals, image DTO cache, etc. Mirrors the same
    // invalidation in addVideosToGallery below. A future optimization could insert into the
    // polymorphic cache shape directly, but the refetch cost is a single HTTP round-trip.
    if (imageDTOs.length > 0) {
      dispatch(galleryApi.util.invalidateTags(['GalleryItemNameList', 'GalleryItemList']));
    }

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
          select: {
            selection: [image_name],
            galleryView: 'images',
          },
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

  const getResultVideoDTOs = async (data: S['InvocationCompleteEvent']): Promise<VideoDTO[]> => {
    const { result } = data;
    const videoDTOs: VideoDTO[] = [];
    for (const [_name, value] of objectEntries(result)) {
      if (isVideoField(value)) {
        const videoDTO = await getVideoDTOSafe(value.video_name);
        if (videoDTO) {
          videoDTOs.push(videoDTO);
        }
      }
    }
    return videoDTOs;
  };

  // Counterpart to addImagesToGallery for VideoField outputs (e.g. Wan 2.2 latents-to-video).
  // Two key differences from the image path:
  //   1. The gallery view uses the polymorphic getGalleryItemNames endpoint and we have no
  //      cheap optimistic-insert here, so we invalidate the GalleryItemNameList/GalleryItemList
  //      tags to force a refetch.
  //   2. The ImageViewerContext's local $progressEvent/$progressImage atoms expect onLoadImage
  //      (DndImage onLoad) to clear them. When auto-switching to a video, the viewer swaps
  //      CurrentImagePreview for CurrentVideoPreview, which unmounts the stale progress overlay
  //      so the stuck "Saving video" spinner goes away on its own.
  const addVideosToGallery = async (data: S['InvocationCompleteEvent']) => {
    if (nodeTypeDenylist.includes(data.invocation.type)) {
      return;
    }

    const videoDTOs = await getResultVideoDTOs(data);
    if (videoDTOs.length === 0) {
      return;
    }

    const nonIntermediate = videoDTOs.filter((v) => !v.is_intermediate);
    if (nonIntermediate.length === 0) {
      return;
    }

    // Force the polymorphic gallery list to refetch so the new video shows up. Note: this is
    // a tag invalidation, not an optimistic insert (the image path has a `insertImageIntoNamesResult`
    // helper, but the polymorphic `GetGalleryItemNamesResult` has a different shape and we don't
    // have an equivalent yet). The viewer selection below applies immediately, so the user sees
    // their video right away; the *gallery grid* scroll-to-selection is delayed by one refetch
    // because `useKeepSelectedImageInView` re-runs when `imageNames` updates and only then can
    // it find the new name in the list. Worth a follow-up if the scroll lag becomes noticeable.
    //
    // The board-affecting helper also invalidates each board's `Board` tag (listAllBoards →
    // video_count and cover_video_name), `BoardVideosTotal`, and `VirtualBoards`, so board
    // counts and cover thumbnails don't lag behind the gallery grid until the next mutation.
    const affectedBoards = [...new Set(nonIntermediate.map((v) => v.board_id ?? 'none'))];
    dispatch(galleryApi.util.invalidateTags(getTagsToInvalidateForBoardAffectingMutation(affectedBoards)));

    const autoSwitch = selectAutoSwitch(getState());
    if (!autoSwitch) {
      return;
    }

    const lastVideoDTO = nonIntermediate.at(-1);
    if (!lastVideoDTO) {
      return;
    }

    const { video_name } = lastVideoDTO;
    const board_id = lastVideoDTO.board_id ?? 'none';

    // Selection is a polymorphic string[]; useGalleryItemDTO discriminates by filename extension.
    const selectedBoardId = selectSelectedBoardId(getState());
    if (board_id !== selectedBoardId) {
      dispatch(
        boardIdSelected({
          boardId: board_id,
          select: {
            selection: [video_name],
            galleryView: 'images',
          },
        })
      );
    } else {
      const galleryView = selectGalleryView(getState());
      if (galleryView !== 'images') {
        dispatch(galleryViewChanged('images'));
      }
      dispatch(imageSelected(video_name));
    }
  };

  const clearCanvasWorkflowIntegrationProcessing = (data: S['InvocationCompleteEvent']) => {
    // Check if this is a canvas workflow integration result
    // Results go to staging area automatically via destination = canvasSessionId
    if (data.origin !== 'canvas_workflow_integration') {
      return;
    }
    // Clear processing state so the modal loading spinner stops
    dispatch(canvasWorkflowIntegrationProcessingCompleted());

    // Check if this invocation produced an image output
    const hasImageOutput = objectEntries(data.result).some(([_name, value]) => {
      return isImageField(value) || isImageFieldCollection(value);
    });

    // Only invalidate if this invocation produced an image - this ensures the staging area
    // gets updated immediately when output images are available, without invalidating on every invocation
    if (hasImageOutput) {
      dispatch(queueApi.util.invalidateTags([{ type: 'SessionQueueItem', id: LIST_ALL_TAG }]));
    }
  };

  return async (data: S['InvocationCompleteEvent']) => {
    log.debug({ data } as JsonObject, `Invocation complete (${data.invocation.type}, ${data.invocation_source_id})`);

    const nodeExecutionState = $nodeExecutionStates.get()[data.invocation_source_id];
    const updatedNodeExecutionState = getUpdatedNodeExecutionStateOnInvocationComplete(
      nodeExecutionState,
      data,
      completedInvocationKeysByItemId
    );

    if (nodeExecutionState && !updatedNodeExecutionState) {
      log.trace(
        { data } as JsonObject,
        `Ignoring duplicate invocation complete (${data.invocation.type}, ${data.invocation_source_id})`
      );
    }

    if (updatedNodeExecutionState) {
      upsertExecutionState(updatedNodeExecutionState.nodeId, updatedNodeExecutionState);
    }

    // Clear canvas workflow integration processing state if needed
    clearCanvasWorkflowIntegrationProcessing(data);

    // Add images to gallery (canvas workflow integration results go to staging area automatically)
    await addImagesToGallery(data);
    await addVideosToGallery(data);

    $lastProgressEvent.set(null);
  };
};

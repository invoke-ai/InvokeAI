import { logger } from 'app/logging/logger';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { canvasWorkflowIntegrationProcessingCompleted } from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import { markNextSelectionAutoSwitched } from 'features/gallery/store/gallerySelectionSource';
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
import { LRUCache } from 'lru-cache';
import type { ApiTagDescription } from 'services/api';
import { api, LIST_ALL_TAG, LIST_TAG } from 'services/api';
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

/**
 * What a completion event still has outstanding.
 *
 * `done` rejects any re-delivery. `retryable` names the outputs whose DTO lookup failed — the ones
 * that never reached the gallery — so a re-delivery can fetch exactly those again. Tracking the
 * missing outputs rather than the event as a whole is what makes a partial failure recoverable: an
 * event-wide flag would either re-run the outputs that did land (double-counting their board totals
 * and optimistic inserts) or abandon the one that did not.
 */
type ProcessedInvocationState =
  | { status: 'done' }
  | {
      status: 'retryable';
      missingNames: ReadonlySet<string>;
      /** How many refetches this event has already had. Carried on the state rather than passed
       * down the call chain so a duplicate delivery resumes the backoff instead of restarting it —
       * a stream of duplicates during an outage would otherwise start a fresh 1s chain each time. */
      attempts: number;
    };

/**
 * When to retry outputs a delivery lost to a failed DTO lookup, and how many times.
 *
 * Recovery used to depend on the server happening to re-deliver the completion event: nothing
 * re-emits one on its own, so an output lost to a transient failure usually stayed lost until the
 * gallery was refetched for some unrelated reason. These are spaced to ride out a brief network
 * blip without hammering a server that is actually down, and bounded because an output that is
 * still missing after ~13 seconds is not coming back from a retry — the image is gone, or the
 * problem is bigger than this handler.
 */
const RETRY_BACKOFF_MS = [1000, 3000, 9000];

/** The completion handler, plus the teardown hook its scheduled refetches require. */
type InvocationCompleteHandler = ((data: S['InvocationCompleteEvent']) => Promise<void>) & {
  /** Ends this handler's session: nothing it has in flight or queued may touch the next one. */
  dispose: () => void;
};

// These nodes are passthrough nodes. They do not add images/videos to the gallery — their
// outputs reference an existing asset — so we must skip the gallery handling for them.
// Without 'video' here, a Video Primitive completing mid-run would invalidate the gallery
// caches and auto-switch the user's selection to the node's *input* video.
const nodeTypeDenylist = ['load_image', 'image', 'video'];

/**
 * Builds the socket event handler for the current user's own invocation complete events. Adds
 * output images to the gallery and/or updates node execution states for the workflow editor.
 * The listener routes other users' events to `buildOnForeignInvocationComplete` instead, so this
 * handler never sees them.
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
  // A duplicate delivery of a completion event must not repeat the work below: re-running the
  // gallery handling double-counts the optimistic board totals and re-dispatches the auto-switch
  // selection. The shared completedInvocationKeysByItemId map cannot detect this — the
  // workflow coordinator pre-marks first-delivery events for non-active workflow items before this
  // handler runs — so the handler tracks what it has processed itself. The invocation id half of
  // the key is the prepared node's per-execution UUID, so keys cannot collide across distinct
  // executions even where item ids restart (in-memory DB); the LRU bounds memory.
  const processedInvocations = new LRUCache<string, ProcessedInvocationState>({ max: 1000 });
  // Deliveries currently fetching, so a duplicate can wait for one instead of being discarded.
  // Entries live only for the duration of a pass; the LRU above is what bounds long-term memory.
  const inFlightDeliveries = new Map<string, Promise<void>>();
  // The pending refetch for each event, so a new pass supersedes it rather than running beside it.
  const scheduledRetries = new Map<string, ReturnType<typeof setTimeout>>();
  // Set when this handler's socket/session goes away. Every await below is a point where the store
  // underneath can have become a different user's, so the flag is checked after each one: clearing
  // the queued timers alone still let a DTO fetch that was already in flight come back and dispatch
  // into — or schedule more work against — the session that replaced this one.
  let isDisposed = false;

  // `retryNames`, when set, restricts this pass to the outputs a previous delivery lost to a failed
  // lookup — everything else already landed and must not be dispatched twice.
  const addImagesToGallery = async (
    data: S['InvocationCompleteEvent'],
    onLookupFailure: (imageName: string) => void,
    retryNames: ReadonlySet<string> | null
  ) => {
    // A retry exists to land an output the gallery lost, not to re-run the handoff around it. By
    // the time a re-delivery arrives the user has had time to select something else, and pulling
    // the selection back would be a worse failure than the one being repaired.
    const isRetry = retryNames !== null;
    if (nodeTypeDenylist.includes(data.invocation.type)) {
      log.trace(`Skipping denylisted node type (${data.invocation.type})`);
      return;
    }

    const fetchedImageDTOs = await getResultImageDTOs(data, onLookupFailure, retryNames);
    if (isDisposed) {
      // The session that asked for these is gone; dispatching now would insert its outputs into
      // whichever one replaced it.
      return;
    }
    // Intermediates never reach the gallery. Dropping them here rather than bailing out of the
    // whole pass on the first one (which is what this used to do) matters now that a delivery's
    // outputs are tracked individually: a sibling abandoned that way is in nobody's missing set,
    // so no re-delivery could ever recover it. Mirrors addVideosToGallery.
    const imageDTOs = fetchedImageDTOs.filter((imageDTO) => !imageDTO.is_intermediate);
    if (imageDTOs.length === 0) {
      return;
    }

    if (isRetry) {
      // A retry lands seconds after the fact, and the optimistic bookkeeping below assumes this
      // delivery is the first thing to know about the output. By now a gallery refetch, or another
      // delivery, may already have inserted it: the name-list insert dedupes, but the board totals
      // are blind increments and would count it twice. Ask the server instead — a retry is rare
      // enough that one authoritative refresh is cheaper than making every optimistic update
      // idempotent against the cache.
      const retriedBoards = [...new Set(imageDTOs.map((imageDTO) => imageDTO.board_id ?? 'none'))];
      dispatch(galleryApi.util.invalidateTags(getTagsToInvalidateForBoardAffectingMutation(retriedBoards)));
      return;
    }

    // For efficiency's sake, we want to minimize the number of dispatches and invalidations we do.
    // We'll keep track of each change we need to make and do them all at once.
    const boardTotalAdditions: Record<string, number> = {};
    const getImageNamesArg = selectGetImageNamesQueryArgs(getState());

    for (const imageDTO of imageDTOs) {
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

    if (!autoSwitch || isRetry) {
      return;
    }

    // Finally, we may need to autoswitch to the new image. We'll only do it for the last image in the list.
    const lastImageDTO = imageDTOs.at(-1);

    if (!lastImageDTO) {
      return;
    }

    const { image_name } = lastImageDTO;
    const board_id = lastImageDTO.board_id ?? 'none';

    // Both branches below auto-switch the selection to this image. Mark it so the viewer's reveal
    // machine can tell the handoff apart from a user's gallery click — this dispatch happens after
    // an async DTO fetch, so it can land after the next generation's first progress event has
    // already reset $isProgressImageResolving, and timing alone cannot distinguish the two.
    markNextSelectionAutoSwitched();

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

  // getImageDTOSafe swallows fetch errors and returns null, which downstream is indistinguishable
  // from "this node produced no image". onLookupFailure separates the two, naming the output that
  // was lost so a re-delivery can fetch just that one again.
  const getResultImageDTOs = async (
    data: S['InvocationCompleteEvent'],
    onLookupFailure: (imageName: string) => void,
    retryNames: ReadonlySet<string> | null
  ): Promise<ImageDTO[]> => {
    const { result } = data;
    const imageDTOs: ImageDTO[] = [];
    const fetched = new Set<string>();
    const fetch = async (imageName: string) => {
      if (isDisposed) {
        // The session that wanted these is gone; issuing more requests would fetch under the
        // replacement session's credentials and write into its RTK cache.
        return;
      }
      if (retryNames !== null && !retryNames.has(imageName)) {
        return;
      }
      // A result can name the same image twice (an image collection concatenates its inputs
      // without deduping). It is still one image: fetching it once keeps its board total counted
      // once, and keeps the retry set — which is keyed by name — from re-admitting an occurrence
      // that already landed.
      if (fetched.has(imageName)) {
        return;
      }
      fetched.add(imageName);
      const imageDTO = await getImageDTOSafe(imageName);
      if (imageDTO) {
        imageDTOs.push(imageDTO);
      } else {
        onLookupFailure(imageName);
      }
    };
    for (const [_name, value] of objectEntries(result)) {
      if (isImageField(value)) {
        await fetch(value.image_name);
      } else if (isImageFieldCollection(value)) {
        for (const imageField of value) {
          await fetch(imageField.image_name);
        }
      }
    }
    return imageDTOs;
  };

  const getResultVideoDTOs = async (
    data: S['InvocationCompleteEvent'],
    onLookupFailure: (videoName: string) => void,
    retryNames: ReadonlySet<string> | null
  ): Promise<VideoDTO[]> => {
    const { result } = data;
    const videoDTOs: VideoDTO[] = [];
    const fetched = new Set<string>();
    for (const [_name, value] of objectEntries(result)) {
      if (isDisposed) {
        break;
      }
      if (isVideoField(value)) {
        if (retryNames !== null && !retryNames.has(value.video_name)) {
          continue;
        }
        if (fetched.has(value.video_name)) {
          continue;
        }
        fetched.add(value.video_name);
        const videoDTO = await getVideoDTOSafe(value.video_name);
        if (videoDTO) {
          videoDTOs.push(videoDTO);
        } else {
          onLookupFailure(value.video_name);
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
  const addVideosToGallery = async (
    data: S['InvocationCompleteEvent'],
    onLookupFailure: (videoName: string) => void,
    retryNames: ReadonlySet<string> | null
  ) => {
    // See addImagesToGallery: a retry does not redo the auto-switch.
    const isRetry = retryNames !== null;
    if (nodeTypeDenylist.includes(data.invocation.type)) {
      return;
    }

    const videoDTOs = await getResultVideoDTOs(data, onLookupFailure, retryNames);
    if (isDisposed) {
      return;
    }
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
    if (!autoSwitch || isRetry) {
      return;
    }

    const lastVideoDTO = nonIntermediate.at(-1);
    if (!lastVideoDTO) {
      return;
    }

    const { video_name } = lastVideoDTO;
    const board_id = lastVideoDTO.board_id ?? 'none';

    // Both branches below auto-switch the selection to this video. Same reasoning as the image
    // path above: the mark is what tells the viewer's reveal machine this is a handoff.
    markNextSelectionAutoSwitched();

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

  /**
   * One pass over a completion event. `retryNames` is null for a first delivery and otherwise names
   * the outputs an earlier delivery lost, restricting this pass to those.
   *
   * Returns the outputs whose lookup failed this time. Bookkeeping is done here, before the
   * returned promise settles, so a duplicate waiting on it always observes the final state.
   */
  const deliver = async (
    data: S['InvocationCompleteEvent'],
    invocationKey: string,
    retryNames: ReadonlySet<string> | null,
    attemptsSoFar: number
  ): Promise<void> => {
    const isRetry = retryNames !== null;
    const missingNames = new Set<string>();
    const onLookupFailure = (itemName: string) => {
      missingNames.add(itemName);
    };

    // A retry redoes the lost gallery work and nothing else: the rest of this handler is not
    // idempotent against a *later* generation, because the canvas processing flag and
    // $lastProgressEvent are global. Re-running them would end the spinner and blank the progress
    // of whatever is running by then.
    if (!isRetry) {
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
    }

    try {
      // Add images to gallery (canvas workflow integration results go to staging area automatically)
      await addImagesToGallery(data, onLookupFailure, retryNames);
      await addVideosToGallery(data, onLookupFailure, retryNames);

      if (!isRetry && !isDisposed) {
        // $lastProgressEvent is module-global, shared across handler sessions: a delivery that was
        // in flight when this session ended must not blank whatever the next session has put there.
        $lastProgressEvent.set(null);
      }
    } catch (error) {
      // Both call sites discard this handler's promise, so a throw here would surface as an
      // unhandled rejection and nothing else. Log it and let the bookkeeping below run: the
      // outputs whose lookups failed are still worth recording as retryable.
      log.error({ data, error } as JsonObject, `Error handling invocation complete: ${String(error)}`);
    } finally {
      if (missingNames.size > 0) {
        processedInvocations.set(invocationKey, { status: 'retryable', missingNames, attempts: attemptsSoFar });
      }
    }
  };

  /**
   * Run one pass and record what it left outstanding, scheduling the next attempt if anything is
   * still missing. `nextAttempt` indexes RETRY_BACKOFF_MS for the retry this pass would schedule;
   * a socket delivery starts at 0, so its first retry waits RETRY_BACKOFF_MS[0].
   */
  const runTrackedDelivery = async (
    data: S['InvocationCompleteEvent'],
    invocationKey: string,
    retryNames: ReadonlySet<string> | null,
    nextAttempt: number
  ): Promise<void> => {
    if (isDisposed) {
      return;
    }

    // This pass supersedes any refetch already queued for the event: leaving that timer armed would
    // run a second chain of attempts alongside this one, doubling the bound RETRY_BACKOFF_MS is
    // supposed to impose.
    const queued = scheduledRetries.get(invocationKey);
    if (queued !== undefined) {
      clearTimeout(queued);
      scheduledRetries.delete(invocationKey);
    }

    // Mark before the awaits below so a duplicate cannot start a second pass over the same outputs
    // while this one is in flight; the handshake in the handler is what lets it retry afterwards.
    processedInvocations.set(invocationKey, { status: 'done' });

    const delivery = deliver(data, invocationKey, retryNames, nextAttempt);
    // Waiters only need to know when the pass finished, not whether it threw.
    const settled = delivery.then(
      () => undefined,
      () => undefined
    );
    inFlightDeliveries.set(invocationKey, settled);
    try {
      await delivery;
    } finally {
      if (inFlightDeliveries.get(invocationKey) === settled) {
        inFlightDeliveries.delete(invocationKey);
      }
    }

    const outcome = processedInvocations.get(invocationKey);
    if (outcome?.status === 'retryable') {
      // nextAttempt is where this pass would resume the backoff; the state carries where the event
      // actually got to, so a duplicate that superseded a queued retry continues the chain rather
      // than restarting it at one second.
      scheduleRetry(data, invocationKey, Math.max(nextAttempt, outcome.attempts), outcome.missingNames);
    }
  };

  const scheduleRetry = (
    data: S['InvocationCompleteEvent'],
    invocationKey: string,
    attempt: number,
    missingNames: ReadonlySet<string>
  ) => {
    if (isDisposed) {
      return;
    }
    const delayMs = RETRY_BACKOFF_MS[attempt];
    if (delayMs === undefined) {
      log.warn(
        { data } as JsonObject,
        `Gave up refetching outputs for ${data.invocation.type} (${data.invocation_source_id}) after ${attempt} attempts`
      );
      // Left 'retryable' on purpose: a re-delivered completion can still recover it for free.
      return;
    }
    // Bare setTimeout, not the window member: this module is unit tested outside a DOM.
    // The names are captured here rather than read back from processedInvocations when the timer
    // fires: that cache is a bounded LRU, and a burst of completions can evict this entry in the
    // meantime. Reading it back would turn eviction into "silently give up on the output", which
    // is the failure the refetch exists to prevent.
    const missingAtSchedule = missingNames;
    const timeout = setTimeout(() => {
      scheduledRetries.delete(invocationKey);
      const state = processedInvocations.get(invocationKey);
      if (state !== undefined && state.status !== 'retryable') {
        // A duplicate delivery got there first, or the outputs landed some other way.
        return;
      }
      void runTrackedDelivery(data, invocationKey, state?.missingNames ?? missingAtSchedule, attempt + 1);
    }, delayMs);
    scheduledRetries.set(invocationKey, timeout);
  };

  const onInvocationComplete: InvocationCompleteHandler = async (data: S['InvocationCompleteEvent']) => {
    log.debug({ data } as JsonObject, `Invocation complete (${data.invocation.type}, ${data.invocation_source_id})`);

    const invocationKey = `${data.item_id}:${data.invocation.id}`;
    const logDuplicate = () => {
      log.trace(
        { data } as JsonObject,
        `Ignoring duplicate invocation complete (${data.invocation.type}, ${data.invocation_source_id})`
      );
    };

    // A duplicate that lands while the first delivery is still fetching waits for it rather than
    // being discarded: if that delivery lost outputs to failed lookups, this duplicate is the only
    // thing that can recover them — nothing re-emits the event on its own. This check must come
    // before the 'done' one, which the in-flight delivery has already written by now.
    //
    // Several duplicates can be waiting here at once. They resume one at a time, and the first to
    // find work marks the event 'done' again before it suspends, so the rest fall through to the
    // duplicate branch rather than starting parallel retries.
    const inFlight = inFlightDeliveries.get(invocationKey);
    if (inFlight) {
      await inFlight;
      if (processedInvocations.get(invocationKey)?.status !== 'retryable') {
        logDuplicate();
        return;
      }
    } else if (processedInvocations.get(invocationKey)?.status === 'done') {
      logDuplicate();
      return;
    }

    const state = processedInvocations.get(invocationKey);
    const retryNames = state?.status === 'retryable' ? state.missingNames : null;
    // A duplicate resumes this event's backoff where it had got to. Starting it over at one second
    // would make the bound meaningless: a stream of duplicates during an outage — exactly when
    // duplicates are likely — would restart the chain indefinitely.
    const resumeAttempt = state?.status === 'retryable' ? state.attempts : 0;
    await runTrackedDelivery(data, invocationKey, retryNames, resumeAttempt);
  };

  /**
   * Drops every pending refetch. Must be called when the socket or the authenticated session goes
   * away: the timers hold a closure over an event from *that* session but dispatch into whatever
   * store is current when they fire, so a logout or account switch inside the retry window would
   * have them fetch the old session's output and insert it into the new one's gallery and board
   * caches.
   */
  onInvocationComplete.dispose = () => {
    isDisposed = true;
    for (const timeout of scheduledRetries.values()) {
      clearTimeout(timeout);
    }
    scheduledRetries.clear();
  };

  return onInvocationComplete;
};

/**
 * Tags that refresh gallery queries after another user's generation completes. Type-level tags
 * because a foreign event's destination board isn't known without fetching image DTOs. RTK Query
 * only refetches invalidated queries that have active subscribers, so an admin actively viewing
 * another user's board stays fresh while idle clients do no work.
 */
export const FOREIGN_GALLERY_REFRESH_TAGS: ApiTagDescription[] = [
  'ImageNameList',
  'BoardImagesTotal',
  'VideoNameList',
  'BoardVideosTotal',
  'GalleryItemList',
  'GalleryItemNameList',
  { type: 'Board', id: LIST_TAG },
  'VirtualBoards',
];

/**
 * Builds the handler for other users' invocation complete events, which admin clients receive via
 * the "admin" socket room. Unlike the user's own completions, these must not fetch image DTOs,
 * edit caches optimistically, auto-switch the gallery, or touch the progress indicator — they
 * only mark gallery caches stale so subscribed queries refetch.
 */
export const buildOnForeignInvocationComplete = (dispatch: AppDispatch) => {
  return (data: S['InvocationCompleteEvent']) => {
    if (nodeTypeDenylist.includes(data.invocation.type)) {
      return;
    }

    // Intermediate images never appear in the gallery, so they cannot change any of the
    // invalidated queries. Saved images inherit this flag from the node, so checking it on the
    // event's invocation payload is equivalent to the own-event path's DTO check — without the
    // fetch. Canvas generations mark most outputs intermediate, so this skips the bulk of
    // foreign completions.
    if (data.invocation.is_intermediate) {
      return;
    }

    const hasGalleryOutput = objectEntries(data.result).some(
      ([_name, value]) => isImageField(value) || isImageFieldCollection(value) || isVideoField(value)
    );
    if (!hasGalleryOutput) {
      return;
    }

    log.trace({ data } as JsonObject, `Foreign invocation complete (${data.invocation.type})`);
    dispatch(api.util.invalidateTags(FOREIGN_GALLERY_REFRESH_TAGS));
  };
};

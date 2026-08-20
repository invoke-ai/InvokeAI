/**
 * Regression test for the polymorphic gallery cache invalidation in
 * ``addImagesToGallery``.
 *
 * The bug: ``onInvocationComplete`` only updated the image-only
 * ``getImageNames`` RTK Query cache via an optimistic insert, but the gallery
 * grid actually reads from the polymorphic ``getGalleryItemNames`` cache. So a
 * freshly-generated image never appeared until the user reloaded the browser,
 * even though it landed in board totals and the per-DTO cache correctly.
 *
 * The fix is a single line that dispatches
 * ``galleryApi.util.invalidateTags(['GalleryItemNameList', 'GalleryItemList'])``
 * after image outputs are processed. This test pins that behavior so a future
 * refactor of the complete handler doesn't silently drop the invalidation.
 */
import type { S } from 'services/api/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// Mock the modules that have heavy side effects on import or do real network work
// when their selectors fire. The mocks return shape-compatible no-ops; we only care
// about the dispatch trace.
vi.mock('services/api/endpoints/images', () => ({
  imagesApi: {
    util: {
      updateQueryData: vi.fn(() => ({ type: 'mock/imagesApi/updateQueryData' })),
      invalidateTags: vi.fn((tags: unknown[]) => ({ type: 'imagesApi/invalidateTags', payload: tags })),
    },
    endpoints: {
      getImageNames: { select: vi.fn(() => () => ({ data: { image_names: [] } })) },
    },
  },
  getImageDTOSafe: vi.fn((image_name: string) =>
    Promise.resolve({
      image_name,
      image_url: `mock://${image_name}`,
      thumbnail_url: `mock://thumb/${image_name}`,
      width: 1024,
      height: 1024,
      is_intermediate: false,
      is_starred: false,
      image_category: 'general',
      image_origin: 'internal',
      has_workflow: false,
      board_id: null,
      created_at: '2026-01-01',
      updated_at: '2026-01-01',
      session_id: 'test-session',
      node_id: 'test-node',
    })
  ),
}));

vi.mock('services/api/endpoints/boards', () => ({
  boardsApi: {
    util: {
      upsertQueryEntries: vi.fn(() => ({ type: 'mock/boardsApi/upsertQueryEntries' })),
      updateQueryData: vi.fn(() => ({ type: 'mock/boardsApi/updateQueryData' })),
    },
    endpoints: {
      getBoardImagesTotal: { select: vi.fn(() => () => ({ data: undefined })) },
    },
  },
}));

vi.mock('services/api/endpoints/queue', () => ({
  queueApi: {
    util: {
      invalidateTags: vi.fn((tags: unknown[]) => ({ type: 'queueApi/invalidateTags', payload: tags })),
    },
  },
}));

vi.mock('services/api/endpoints/videos', () => ({
  getVideoDTOSafe: vi.fn(() => Promise.resolve(null)),
}));

vi.mock('features/gallery/store/gallerySelectors', () => ({
  selectAutoSwitch: vi.fn(() => false),
  selectGalleryView: vi.fn(() => 'images'),
  selectGetImageNamesQueryArgs: vi.fn(() => ({
    board_id: 'none',
    categories: ['general'],
    search_term: '',
    order_dir: 'DESC',
    starred_first: true,
    is_intermediate: false,
  })),
  selectListBoardsQueryArgs: vi.fn(() => ({
    order_by: 'created_at',
    direction: 'DESC',
  })),
  selectSelectedBoardId: vi.fn(() => 'none'),
}));

vi.mock('features/gallery/store/gallerySlice', () => ({
  boardIdSelected: vi.fn(() => ({ type: 'mock/boardIdSelected' })),
  galleryViewChanged: vi.fn(() => ({ type: 'mock/galleryViewChanged' })),
  imageSelected: vi.fn(() => ({ type: 'mock/imageSelected' })),
}));

vi.mock('features/controlLayers/store/canvasWorkflowIntegrationSlice', () => ({
  canvasWorkflowIntegrationProcessingCompleted: vi.fn(() => ({ type: 'mock/canvasComplete' })),
}));

vi.mock('features/nodes/hooks/useNodeExecutionState', () => ({
  $nodeExecutionStates: { get: vi.fn(() => ({})) },
  upsertExecutionState: vi.fn(),
}));

vi.mock('services/events/nodeExecutionState', () => ({
  getUpdatedNodeExecutionStateOnInvocationComplete: vi.fn(() => null),
}));

// Mocked so a scheduled refetch that blunders into work it should have skipped is observable:
// deliver() catches its own throw and logs it, which is otherwise invisible from out here.
vi.mock('app/logging/logger', () => {
  const log = { debug: vi.fn(), trace: vi.fn(), warn: vi.fn(), error: vi.fn() };
  return { logger: () => log };
});

vi.mock('services/events/stores', () => ({
  $lastProgressEvent: { set: vi.fn() },
}));

// Import AFTER the mocks above are declared (vi.mock is hoisted; explicit ordering here
// is for the human reader).
import { logger } from 'app/logging/logger';
import { autoSwitchedImages } from 'features/gallery/store/autoSwitchedImages';
import { selectAutoSwitch } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { getImageDTOSafe } from 'services/api/endpoints/images';
import { getVideoDTOSafe } from 'services/api/endpoints/videos';
import { $lastProgressEvent } from 'services/events/stores';

import {
  buildOnForeignInvocationComplete,
  buildOnInvocationComplete,
  FOREIGN_GALLERY_REFRESH_TAGS,
} from './onInvocationComplete';

// Build a synthetic InvocationCompleteEvent whose result contains a single ImageField output.
// The runtime ``isImageField`` discriminator matches on ``type === 'image_output'``.
const buildImageCompleteEvent = (): S['InvocationCompleteEvent'] =>
  ({
    queue_id: 'default',
    item_id: 1,
    batch_id: 'batch-1',
    origin: 'workflows',
    destination: 'gallery',
    user_id: 'user-1',
    session_id: 'session-1',
    invocation_source_id: 'node-1',
    invocation: {
      id: 'prepared-node-1',
      // Not in nodeTypeDenylist (which contains 'load_image', 'image') — so the handler
      // will proceed to extract image DTOs.
      type: 'add',
    },
    // ``result`` is the node's OutputType serialized as a flat key→value map.
    // ``isImageField`` accepts any object with a non-empty ``image_name`` string,
    // which is what the ``image`` output field unwraps to.
    result: {
      image: { image_name: 'fresh-image.png' },
      width: 1024,
      height: 1024,
    },
  }) as unknown as S['InvocationCompleteEvent'];

describe('onInvocationComplete polymorphic gallery cache', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // clearAllMocks resets calls, not implementations — restore the factory default so a test that
    // turns auto-switch on cannot leak it into the next one.
    vi.mocked(selectAutoSwitch).mockReturnValue(false);
    autoSwitchedImages.settle(null);
  });

  it('invalidates GalleryItemNameList + GalleryItemList when an image output completes', async () => {
    const dispatched: unknown[] = [];
    const dispatch = vi.fn((action: unknown) => {
      dispatched.push(action);
      // RTK Query thunks return unsubscribe promises; the handler does not chain on the
      // return value of the invalidate dispatch, so we can synchronously return a stub.
      return { unwrap: () => Promise.resolve(undefined) };
    });
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    await handler(buildImageCompleteEvent());

    // The handler emits many actions; the regression-critical one is the polymorphic
    // gallery tag invalidation. We identify it by its payload — the real
    // ``galleryApi.util.invalidateTags`` produces an action with this exact payload.
    const galleryInvalidation = dispatched.find((action): action is { type: string; payload: string[] } => {
      if (!action || typeof action !== 'object') {
        return false;
      }
      const payload = (action as { payload?: unknown }).payload;
      if (!Array.isArray(payload)) {
        return false;
      }
      return payload.includes('GalleryItemNameList') && payload.includes('GalleryItemList');
    });

    expect(galleryInvalidation, 'addImagesToGallery must invalidate the polymorphic gallery cache').toBeDefined();
  });

  it('does not invalidate the polymorphic gallery cache for denylisted node types', async () => {
    const dispatched: unknown[] = [];
    const dispatch = vi.fn((action: unknown) => {
      dispatched.push(action);
      return { unwrap: () => Promise.resolve(undefined) };
    });
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    // ``image`` is in the nodeTypeDenylist (passthrough node — doesn't add to gallery).
    const denylisted = buildImageCompleteEvent();
    denylisted.invocation.type = 'image';

    await handler(denylisted);

    const galleryInvalidation = dispatched.find((action): action is { type: string; payload: string[] } => {
      if (!action || typeof action !== 'object') {
        return false;
      }
      const payload = (action as { payload?: unknown }).payload;
      return Array.isArray(payload) && payload.includes('GalleryItemNameList');
    });
    expect(galleryInvalidation, 'denylisted passthrough nodes must not trigger a gallery refetch').toBeUndefined();
  });

  it('invalidates board tags/totals in addition to the gallery cache when a video output completes', async () => {
    // A generated video landing on a board must also refresh that board's DTO (video_count,
    // cover_video_name via the ``Board`` tag), its ``BoardVideosTotal``, and the virtual board
    // groupings — otherwise the boards list shows stale counts/covers until some unrelated
    // mutation happens to refetch them.
    vi.mocked(getVideoDTOSafe).mockResolvedValueOnce({
      video_name: 'fresh-video.mp4',
      video_url: 'mock://fresh-video.mp4',
      thumbnail_url: 'mock://thumb/fresh-video.mp4',
      width: 832,
      height: 480,
      duration_seconds: 3.4,
      frame_count: 81,
      is_intermediate: false,
      is_starred: false,
      board_id: 'board-123',
      created_at: '2026-01-01',
      updated_at: '2026-01-01',
      session_id: 'test-session',
      node_id: 'test-node',
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    const dispatched: unknown[] = [];
    const dispatch = vi.fn((action: unknown) => {
      dispatched.push(action);
      return { unwrap: () => Promise.resolve(undefined) };
    });
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    const videoEvent = buildImageCompleteEvent();
    videoEvent.invocation.type = 'wan_l2v';
    // ``isVideoField`` accepts any object with a non-empty ``video_name`` string.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (videoEvent as any).result = { video: { video_name: 'fresh-video.mp4' } };

    await handler(videoEvent);

    const galleryInvalidation = dispatched.find((action): action is { type: string; payload: unknown[] } => {
      if (!action || typeof action !== 'object') {
        return false;
      }
      const payload = (action as { payload?: unknown }).payload;
      return Array.isArray(payload) && payload.includes('GalleryItemNameList');
    });

    expect(galleryInvalidation, 'video completion must invalidate the polymorphic gallery cache').toBeDefined();
    // The same invalidation must cover the board caches for the video's board.
    expect(galleryInvalidation?.payload).toContainEqual({ type: 'Board', id: 'board-123' });
    expect(galleryInvalidation?.payload).toContainEqual({ type: 'BoardVideosTotal', id: 'board-123' });
    expect(galleryInvalidation?.payload).toContain('VirtualBoards');
  });

  it('processes each completion event exactly once — a duplicate delivery does no gallery work', async () => {
    // Re-running the gallery handling on a duplicate double-counts the optimistic board totals and
    // re-records the auto-switch marker after it was consumed, which suppresses a later genuine
    // gallery click on that image.
    const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    await handler(buildImageCompleteEvent());
    const dispatchCountAfterFirst = dispatch.mock.calls.length;
    expect(getImageDTOSafe).toHaveBeenCalledTimes(1);

    await handler(buildImageCompleteEvent());
    expect(getImageDTOSafe).toHaveBeenCalledTimes(1);
    expect(dispatch.mock.calls.length).toBe(dispatchCountAfterFirst);
  });

  it('rejects a duplicate that arrives while the first delivery is still awaiting its DTO fetch', async () => {
    // The gallery work awaits a DTO fetch, so a duplicate can land mid-flight. The handler must
    // mark the event as processed before the first await, not after.
    const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    await Promise.all([handler(buildImageCompleteEvent()), handler(buildImageCompleteEvent())]);
    expect(getImageDTOSafe).toHaveBeenCalledTimes(1);
  });

  it('records a duplicated video completion once, so a later reselection of that video reveals', async () => {
    // The duplicate delivery used to re-run addVideosToGallery in full: a second auto-switch
    // marker record and a second selection dispatch. The dedupe must stop the whole re-run.
    autoSwitchedImages.settle(null); // the marker is a module singleton; start from empty
    vi.mocked(selectAutoSwitch).mockReturnValueOnce(true);
    vi.mocked(getVideoDTOSafe).mockResolvedValueOnce({
      video_name: 'fresh-video.mp4',
      video_url: 'mock://fresh-video.mp4',
      thumbnail_url: 'mock://thumb/fresh-video.mp4',
      width: 832,
      height: 480,
      duration_seconds: 3.4,
      frame_count: 81,
      is_intermediate: false,
      is_starred: false,
      board_id: null,
      created_at: '2026-01-01',
      updated_at: '2026-01-01',
      session_id: 'test-session',
      node_id: 'test-node',
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    const videoEvent = buildImageCompleteEvent();
    videoEvent.invocation.type = 'wan_l2v';
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (videoEvent as any).result = { video: { video_name: 'fresh-video.mp4' } };

    await handler(videoEvent);
    await handler(videoEvent);

    expect(getVideoDTOSafe).toHaveBeenCalledTimes(1);
    expect(imageSelected).toHaveBeenCalledTimes(1);

    // The rest of JPPhoto's sequence, against the real marker singleton: the auto-switched video
    // renders (consume), the user selects another item, then re-selects the video — which must
    // NOT read as an auto-switch, or the click is dead under the progress overlay.
    autoSwitchedImages.settle('fresh-video.mp4');
    expect(autoSwitchedImages.consume('fresh-video.mp4')).toBe(true);
    autoSwitchedImages.settle('some-other-item.png');
    autoSwitchedImages.settle('fresh-video.mp4');
    expect(autoSwitchedImages.consume('fresh-video.mp4')).toBe(false);
  });

  it('lets a re-delivery retry when the first attempt lost its DTO to a transient failure', async () => {
    // getImageDTOSafe swallows fetch errors and returns null, so a transient failure silently
    // drops the image from the gallery. Keeping the dedupe key in that case would make the loss
    // permanent — the redelivery that could fix it is turned away as a duplicate.
    vi.mocked(getImageDTOSafe).mockResolvedValueOnce(null);

    const dispatched: unknown[] = [];
    const dispatch = vi.fn((action: unknown) => {
      dispatched.push(action);
      return { unwrap: () => Promise.resolve(undefined) };
    });
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    await handler(buildImageCompleteEvent());
    expect(
      dispatched.some((action) => {
        const payload = (action as { payload?: unknown }).payload;
        return Array.isArray(payload) && payload.includes('GalleryItemNameList');
      }),
      'a failed lookup produces no gallery work'
    ).toBe(false);

    await handler(buildImageCompleteEvent());
    expect(getImageDTOSafe).toHaveBeenCalledTimes(2);
    expect(
      dispatched.some((action) => {
        const payload = (action as { payload?: unknown }).payload;
        return Array.isArray(payload) && payload.includes('GalleryItemNameList');
      }),
      'the retry must do the gallery work the first delivery lost'
    ).toBe(true);
  });

  it('retries only the output whose lookup failed, leaving the ones that landed alone', async () => {
    // One of two image lookups fails. The successful one's board totals and optimistic insert have
    // already gone out, so a re-delivery must not touch it — but the lost one is only recoverable
    // here, since nothing re-emits the event on its own.
    vi.mocked(getImageDTOSafe).mockResolvedValueOnce(null);

    const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    const twoImages = buildImageCompleteEvent();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (twoImages as any).result = {
      image_1: { image_name: 'first.png' },
      image_2: { image_name: 'second.png' },
    };

    await handler(twoImages);
    expect(getImageDTOSafe).toHaveBeenCalledTimes(2);

    await handler(twoImages);
    // Exactly one more lookup, and it is the one that failed.
    expect(getImageDTOSafe).toHaveBeenCalledTimes(3);
    expect(vi.mocked(getImageDTOSafe).mock.calls.at(-1)?.[0]).toBe('first.png');

    // Once everything has landed, the event is closed again.
    await handler(twoImages);
    expect(getImageDTOSafe).toHaveBeenCalledTimes(3);
  });

  it('retries a failed lookup inside an image collection', async () => {
    // Collections are where partial failure is actually plausible: the user sees most of a batch
    // and silently misses one image, its auto-switch, and its board count.
    vi.mocked(getImageDTOSafe).mockResolvedValueOnce(null);

    const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    const collection = buildImageCompleteEvent();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (collection as any).result = {
      collection: [{ image_name: 'batch-1.png' }, { image_name: 'batch-2.png' }, { image_name: 'batch-3.png' }],
    };

    await handler(collection);
    expect(getImageDTOSafe).toHaveBeenCalledTimes(3);

    await handler(collection);
    expect(getImageDTOSafe).toHaveBeenCalledTimes(4);
    expect(vi.mocked(getImageDTOSafe).mock.calls.at(-1)?.[0]).toBe('batch-1.png');
  });

  it('lets a duplicate that overlapped a lost delivery become the retry', async () => {
    // The duplicate arrives while the first delivery is still fetching, so it cannot be told yet
    // that the fetch will fail. Rejecting it outright strands the output: there is no third event.
    let resolveFirstLookup: (value: null) => void = () => {};
    vi.mocked(getImageDTOSafe).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveFirstLookup = resolve;
        })
    );

    const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    const event = buildImageCompleteEvent();
    const first = handler(event);
    const duplicate = handler(event);

    // The first delivery's only lookup fails, losing the image.
    resolveFirstLookup(null);
    await Promise.all([first, duplicate]);

    // The duplicate picked the work back up rather than being discarded.
    expect(getImageDTOSafe).toHaveBeenCalledTimes(2);
    expect(vi.mocked(getImageDTOSafe).mock.calls.at(-1)?.[0]).toBe('fresh-image.png');
  });

  it('marks the selection it auto-switches to, so the viewer does not reveal it as a user click', async () => {
    // Without the marker the finished image flashes over the next generation's live preview for
    // two seconds — the regression this PR exists to fix. The marker is the only signal that can
    // tell the handoff from a click, since it lands after the next generation's first progress
    // event has already reset the timing-based guard.
    vi.mocked(selectAutoSwitch).mockReturnValue(true);
    autoSwitchedImages.settle(null);

    const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    await handler(buildImageCompleteEvent());

    expect(autoSwitchedImages.consume('fresh-image.png')).toBe(true);
  });

  it('does not mark anything when auto-switch is off — nothing is selected to be revealed', async () => {
    vi.mocked(selectAutoSwitch).mockReturnValue(false);
    autoSwitchedImages.settle(null);

    const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    await handler(buildImageCompleteEvent());

    expect(autoSwitchedImages.consume('fresh-image.png')).toBe(false);
  });

  it('retries only the image half of a mixed result, not the video that landed', async () => {
    // The video lookup succeeded, so its board invalidation and auto-switch already went out; a
    // re-delivery re-running them would invalidate twice and move the selection a second time.
    vi.mocked(selectAutoSwitch).mockReturnValue(true);
    vi.mocked(getImageDTOSafe).mockResolvedValueOnce(null);
    vi.mocked(getVideoDTOSafe).mockResolvedValueOnce({
      video_name: 'fresh-video.mp4',
      video_url: 'mock://fresh-video.mp4',
      thumbnail_url: 'mock://thumb/fresh-video.mp4',
      is_intermediate: false,
      is_starred: false,
      board_id: 'board-123',
      created_at: '2026-01-01',
      updated_at: '2026-01-01',
      session_id: 'test-session',
      node_id: 'test-node',
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    const mixed = buildImageCompleteEvent();
    mixed.invocation.type = 'wan_l2v';
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (mixed as any).result = {
      image: { image_name: 'lost-image.png' },
      video: { video_name: 'fresh-video.mp4' },
    };

    await handler(mixed);
    expect(getVideoDTOSafe).toHaveBeenCalledTimes(1);
    expect(getImageDTOSafe).toHaveBeenCalledTimes(1);

    // The re-delivery refetches the lost image and leaves the video alone.
    await handler(mixed);
    expect(getImageDTOSafe).toHaveBeenCalledTimes(2);
    expect(getVideoDTOSafe).toHaveBeenCalledTimes(1);
  });

  it('redoes only the gallery work on a retry, not the global side effects', async () => {
    // The canvas processing flag and $lastProgressEvent are global: by the time a re-delivery
    // arrives the user may have started another run, and clearing them again would stop that run's
    // spinner and blank its progress.
    vi.mocked(getImageDTOSafe).mockResolvedValueOnce(null);

    const dispatched: unknown[] = [];
    const dispatch = vi.fn((action: unknown) => {
      dispatched.push(action);
      return { unwrap: () => Promise.resolve(undefined) };
    });
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    const canvasEvent = buildImageCompleteEvent();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (canvasEvent as any).origin = 'canvas_workflow_integration';

    await handler(canvasEvent);
    const canvasClears = () => dispatched.filter((a) => (a as { type?: string }).type === 'mock/canvasComplete').length;
    expect(canvasClears()).toBe(1);
    expect($lastProgressEvent.set).toHaveBeenCalledTimes(1);

    // The retry must redo the lost gallery fetch...
    await handler(canvasEvent);
    expect(getImageDTOSafe).toHaveBeenCalledTimes(2);
    // ...and nothing else.
    expect(canvasClears()).toBe(1);
    expect($lastProgressEvent.set).toHaveBeenCalledTimes(1);
  });

  it('retries the lost output alongside an intermediate one, without redoing the intermediate', async () => {
    // An intermediate image never reaches the gallery, so there is nothing to redo for it — but the
    // lookup that failed beside it must still be recoverable.
    vi.mocked(getImageDTOSafe)
      .mockResolvedValueOnce(null)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .mockResolvedValueOnce({ image_name: 'intermediate.png', is_intermediate: true, board_id: null } as any);

    const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    const twoImages = buildImageCompleteEvent();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (twoImages as any).result = {
      image_1: { image_name: 'lost.png' },
      image_2: { image_name: 'intermediate.png' },
    };

    await handler(twoImages);
    expect(getImageDTOSafe).toHaveBeenCalledTimes(2);

    // Only the lost output is refetched — the intermediate one never had gallery work to redo.
    await handler(twoImages);
    expect(getImageDTOSafe).toHaveBeenCalledTimes(3);
    expect(vi.mocked(getImageDTOSafe).mock.calls.at(-1)?.[0]).toBe('lost.png');
  });

  it('serializes several duplicates waiting on one lost delivery into a single retry', async () => {
    // All three duplicates are parked on the same in-flight delivery. When it fails they all wake
    // up; only one may pick the work back up, or the retried output lands two or three times over.
    let resolveFirstLookup: (value: null) => void = () => {};
    vi.mocked(getImageDTOSafe).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveFirstLookup = resolve;
        })
    );

    const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    const event = buildImageCompleteEvent();
    const deliveries = [handler(event), handler(event), handler(event), handler(event)];

    resolveFirstLookup(null);
    await Promise.all(deliveries);

    // One failed lookup plus exactly one retry.
    expect(getImageDTOSafe).toHaveBeenCalledTimes(2);
  });

  it('counts an image named twice in one result once, and once again on retry', async () => {
    // An image collection concatenates its inputs without deduping, so the same name can appear
    // twice. It is one image: counting it twice inflates the board total, and on a retry a
    // name-keyed missing set would re-admit the occurrence that already landed.
    vi.mocked(selectAutoSwitch).mockReturnValue(true);
    vi.mocked(getImageDTOSafe).mockResolvedValueOnce(null);

    const dispatched: unknown[] = [];
    const dispatch = vi.fn((action: unknown) => {
      dispatched.push(action);
      return { unwrap: () => Promise.resolve(undefined) };
    });
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    const duplicated = buildImageCompleteEvent();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (duplicated as any).result = {
      collection: [{ image_name: 'dup.png' }, { image_name: 'dup.png' }],
    };

    // Pass 1: one lookup for the repeated name, and it fails, so nothing lands.
    await handler(duplicated);
    expect(getImageDTOSafe).toHaveBeenCalledTimes(1);

    // Pass 2: still one lookup, and the image is counted exactly once.
    await handler(duplicated);
    expect(getImageDTOSafe).toHaveBeenCalledTimes(2);

    const boardTotalUpdates = dispatched.filter(
      (action) => (action as { type?: string }).type === 'mock/boardsApi/upsertQueryEntries'
    );
    expect(boardTotalUpdates).toHaveLength(1);
  });

  it('keeps a non-intermediate sibling of an intermediate output', async () => {
    // This used to bail out of the whole pass on the first intermediate, abandoning siblings that
    // belong in the gallery — and with per-output retry such a sibling is in nobody's missing set,
    // so no re-delivery could recover it.
    // mockResolvedValueOnce, not mockImplementation: clearAllMocks resets calls but not
    // implementations, so a persistent one would leak into every test after this.
    const dto = (imageName: string, isIntermediate: boolean) =>
      ({
        image_name: imageName,
        image_url: `mock://${imageName}`,
        thumbnail_url: `mock://thumb/${imageName}`,
        is_intermediate: isIntermediate,
        is_starred: false,
        image_category: 'general',
        board_id: null,
        created_at: '2026-01-01',
        updated_at: '2026-01-01',
        session_id: 'test-session',
        node_id: 'test-node',
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      }) as any;
    vi.mocked(getImageDTOSafe)
      .mockResolvedValueOnce(dto('intermediate.png', true))
      .mockResolvedValueOnce(dto('keeper.png', false));

    const dispatched: unknown[] = [];
    const dispatch = vi.fn((action: unknown) => {
      dispatched.push(action);
      return { unwrap: () => Promise.resolve(undefined) };
    });
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    const mixed = buildImageCompleteEvent();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (mixed as any).result = {
      collection: [{ image_name: 'intermediate.png' }, { image_name: 'keeper.png' }],
    };

    await handler(mixed);

    const galleryInvalidation = dispatched.find((action) => {
      const payload = (action as { payload?: unknown }).payload;
      return Array.isArray(payload) && payload.includes('GalleryItemNameList');
    });
    expect(galleryInvalidation, 'the non-intermediate sibling must reach the gallery').toBeDefined();
  });

  it('does not move the selection again when a retry lands the lost output', async () => {
    // The re-delivery can arrive long after the user has selected something else.
    vi.mocked(selectAutoSwitch).mockReturnValue(true);
    vi.mocked(getImageDTOSafe).mockResolvedValueOnce(null);

    const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    await handler(buildImageCompleteEvent());
    expect(imageSelected).not.toHaveBeenCalled();

    await handler(buildImageCompleteEvent());
    expect(getImageDTOSafe).toHaveBeenCalledTimes(2);
    expect(imageSelected, 'a retry lands the image without re-running the handoff').not.toHaveBeenCalled();
  });

  it('logs rather than rejecting when the gallery work throws', async () => {
    // Both call sites discard this handler's promise, so a rejection would surface only as an
    // unhandled rejection.
    const dispatch = vi.fn(() => {
      throw new Error('dispatch exploded');
    });
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    await expect(handler(buildImageCompleteEvent())).resolves.toBeUndefined();
  });

  it('refetches a lost output on its own, without waiting for a duplicate delivery', async () => {
    // Nothing re-emits a completion event, so before this the recovery path only ran if the server
    // happened to send the event twice — which usually meant the output stayed missing.
    vi.useFakeTimers();
    try {
      vi.mocked(getImageDTOSafe).mockResolvedValueOnce(null);

      const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
      const getState = vi.fn(() => ({}));
      const handler = buildOnInvocationComplete(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        getState as any,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        dispatch as any,
        new Map()
      );

      await handler(buildImageCompleteEvent());
      expect(getImageDTOSafe).toHaveBeenCalledTimes(1);

      await vi.advanceTimersByTimeAsync(1000);
      expect(getImageDTOSafe, 'the first backoff step refetches the missing name').toHaveBeenCalledTimes(2);

      // It succeeded this time, so nothing further is scheduled.
      await vi.advanceTimersByTimeAsync(60_000);
      expect(getImageDTOSafe).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it('gives up after a bounded number of refetches', async () => {
    // A permanently missing output must not retry forever.
    vi.useFakeTimers();
    try {
      vi.mocked(getImageDTOSafe).mockResolvedValue(null);

      const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
      const getState = vi.fn(() => ({}));
      const handler = buildOnInvocationComplete(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        getState as any,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        dispatch as any,
        new Map()
      );

      await handler(buildImageCompleteEvent());
      await vi.advanceTimersByTimeAsync(120_000);

      // The delivery plus the three backoff steps, and no more.
      expect(getImageDTOSafe).toHaveBeenCalledTimes(4);
    } finally {
      vi.mocked(getImageDTOSafe).mockReset();
      vi.useRealTimers();
    }
  });

  it('drops a scheduled refetch when a duplicate delivery recovers the output first', async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(getImageDTOSafe).mockResolvedValueOnce(null);

      const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
      const getState = vi.fn(() => ({}));
      const handler = buildOnInvocationComplete(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        getState as any,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        dispatch as any,
        new Map()
      );

      await handler(buildImageCompleteEvent());
      // A re-delivery lands before the first backoff step and does the work.
      await handler(buildImageCompleteEvent());
      expect(getImageDTOSafe).toHaveBeenCalledTimes(2);

      await vi.advanceTimersByTimeAsync(60_000);
      expect(getImageDTOSafe, 'the scheduled retry finds nothing left to do').toHaveBeenCalledTimes(2);
      // ...and it works that out before starting a pass, rather than failing inside one.
      expect(logger('events').error).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not run a second chain of refetches when a duplicate delivery lands', async () => {
    // Each pass supersedes the refetch already queued for that event; two chains would run to
    // exhaustion side by side and double the bound the backoff is there to impose.
    vi.useFakeTimers();
    try {
      vi.mocked(getImageDTOSafe).mockResolvedValue(null);

      const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
      const getState = vi.fn(() => ({}));
      const handler = buildOnInvocationComplete(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        getState as any,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        dispatch as any,
        new Map()
      );

      await handler(buildImageCompleteEvent());
      await vi.advanceTimersByTimeAsync(500);
      // A duplicate arrives mid-backoff and takes the work over.
      await handler(buildImageCompleteEvent());
      await vi.advanceTimersByTimeAsync(120_000);

      // Two deliveries plus one chain of three attempts — not two chains.
      expect(getImageDTOSafe).toHaveBeenCalledTimes(5);
    } finally {
      vi.mocked(getImageDTOSafe).mockReset();
      vi.useRealTimers();
    }
  });

  it('still processes distinct invocations of the same queue item', async () => {
    const dispatch = vi.fn(() => ({ unwrap: () => Promise.resolve(undefined) }));
    const getState = vi.fn(() => ({}));

    const handler = buildOnInvocationComplete(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getState as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatch as any,
      new Map()
    );

    await handler(buildImageCompleteEvent());
    const secondNode = buildImageCompleteEvent();
    secondNode.invocation.id = 'prepared-node-2';
    await handler(secondNode);
    expect(getImageDTOSafe).toHaveBeenCalledTimes(2);
  });
});

describe('buildOnForeignInvocationComplete', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const buildForeignHandler = () => {
    const dispatched: unknown[] = [];
    const dispatch = vi.fn((action: unknown) => {
      dispatched.push(action);
    });
    const handler = buildOnForeignInvocationComplete(dispatch as never);
    return { handler, dispatch, dispatched };
  };

  it('refreshes gallery caches via tag invalidation only — no DTO fetches, no selection changes', () => {
    const { handler, dispatch, dispatched } = buildForeignHandler();

    handler(buildImageCompleteEvent());

    // Exactly one dispatch: the tag invalidation. No optimistic cache edits, no board/image
    // selection, no progress clear.
    expect(dispatch).toHaveBeenCalledTimes(1);
    const action = dispatched[0] as { payload?: unknown };
    expect(action.payload).toEqual(FOREIGN_GALLERY_REFRESH_TAGS);
    // The WAN gallery reads polymorphic (image+video) caches, so the foreign refresh must
    // cover both media types plus board totals.
    expect(FOREIGN_GALLERY_REFRESH_TAGS).toEqual(
      expect.arrayContaining([
        'ImageNameList',
        'BoardImagesTotal',
        'VideoNameList',
        'BoardVideosTotal',
        'GalleryItemList',
        'GalleryItemNameList',
        'VirtualBoards',
      ])
    );
    expect(getImageDTOSafe).not.toHaveBeenCalled();
    expect(getVideoDTOSafe).not.toHaveBeenCalled();
  });

  it('invalidates gallery caches for foreign video outputs', () => {
    const { handler, dispatch } = buildForeignHandler();

    const videoEvent = buildImageCompleteEvent();
    videoEvent.invocation.type = 'wan_l2v';
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (videoEvent as any).result = { video: { video_name: 'foreign-video.mp4' } };

    handler(videoEvent);

    expect(dispatch).toHaveBeenCalledTimes(1);
  });

  it('does nothing for results without gallery outputs', () => {
    const { handler, dispatch } = buildForeignHandler();

    const latentsEvent = buildImageCompleteEvent();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (latentsEvent as any).result = { latents: { latents_name: 'latents-1' } };

    handler(latentsEvent);

    expect(dispatch).not.toHaveBeenCalled();
  });

  it('does nothing for denylisted passthrough node types', () => {
    const { handler, dispatch } = buildForeignHandler();

    const denylisted = buildImageCompleteEvent();
    denylisted.invocation.type = 'image';

    handler(denylisted);

    expect(dispatch).not.toHaveBeenCalled();
  });

  it('does nothing for intermediate outputs, which never appear in the gallery', () => {
    const { handler, dispatch } = buildForeignHandler();

    const intermediate = buildImageCompleteEvent();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (intermediate.invocation as any).is_intermediate = true;

    handler(intermediate);

    expect(dispatch).not.toHaveBeenCalled();
  });
});

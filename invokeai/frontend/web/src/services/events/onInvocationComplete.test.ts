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

vi.mock('services/events/stores', () => ({
  $lastProgressEvent: { set: vi.fn() },
}));

// Import AFTER the mocks above are declared (vi.mock is hoisted; explicit ordering here
// is for the human reader).
import { getImageDTOSafe } from 'services/api/endpoints/images';
import { getVideoDTOSafe } from 'services/api/endpoints/videos';

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

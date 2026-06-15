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
import { buildOnInvocationComplete } from './onInvocationComplete';

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
});

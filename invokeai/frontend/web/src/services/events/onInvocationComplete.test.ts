import { boardIdSelected, imageSelected } from 'features/gallery/store/gallerySlice';
import { api } from 'services/api';
import { getImageDTOSafe } from 'services/api/endpoints/images';
import type { ImageDTO, S } from 'services/api/types';
import { $lastProgressEvent } from 'services/events/stores';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  buildOnForeignInvocationComplete,
  buildOnInvocationComplete,
  FOREIGN_GALLERY_REFRESH_TAGS,
} from './onInvocationComplete';

vi.mock('app/logging/logger', () => ({
  logger: () => ({
    debug: vi.fn(),
    trace: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  }),
}));

vi.mock('services/api', () => ({
  LIST_TAG: 'LIST',
  LIST_ALL_TAG: 'LIST_ALL',
  api: {
    util: {
      invalidateTags: vi.fn((tags) => ({ type: 'api/invalidateTags', payload: tags })),
    },
  },
}));

vi.mock('services/api/endpoints/images', () => ({
  getImageDTOSafe: vi.fn(),
  imagesApi: {
    util: {
      updateQueryData: vi.fn(() => ({ type: 'imagesApi/updateQueryData' })),
      invalidateTags: vi.fn(() => ({ type: 'imagesApi/invalidateTags' })),
    },
  },
}));

vi.mock('services/api/endpoints/boards', () => ({
  boardsApi: {
    endpoints: {
      getBoardImagesTotal: {
        select: vi.fn(() => () => ({ data: undefined })),
      },
    },
    util: {
      upsertQueryEntries: vi.fn(() => ({ type: 'boardsApi/upsertQueryEntries' })),
      updateQueryData: vi.fn(() => ({ type: 'boardsApi/updateQueryData' })),
    },
  },
}));

vi.mock('services/api/endpoints/queue', () => ({
  queueApi: {
    util: {
      invalidateTags: vi.fn(() => ({ type: 'queueApi/invalidateTags' })),
    },
  },
}));

vi.mock('features/gallery/store/gallerySelectors', () => ({
  selectAutoSwitch: () => true,
  selectGalleryView: () => 'images',
  selectGetImageNamesQueryArgs: () => ({ search_term: '', starred_first: true, order_dir: 'DESC' }),
  selectListBoardsQueryArgs: () => ({}),
  selectSelectedBoardId: () => 'other-board',
}));

vi.mock('features/nodes/hooks/useNodeExecutionState', () => ({
  $nodeExecutionStates: { get: () => ({}) },
  upsertExecutionState: vi.fn(),
}));

vi.mock('features/controlLayers/store/canvasWorkflowIntegrationSlice', () => ({
  canvasWorkflowIntegrationProcessingCompleted: vi.fn(() => ({
    type: 'canvasWorkflowIntegration/processingCompleted',
  })),
}));

vi.mock('services/events/nodeExecutionState', () => ({
  getUpdatedNodeExecutionStateOnInvocationComplete: vi.fn(() => undefined),
}));

vi.mock('services/events/stores', () => ({
  $lastProgressEvent: { set: vi.fn() },
}));

const imageDTO = {
  image_name: 'img-1.png',
  board_id: 'user-board',
  is_intermediate: false,
  image_category: 'general',
} as ImageDTO;

const buildEvent = (overrides: Record<string, unknown> = {}): S['InvocationCompleteEvent'] =>
  ({
    item_id: 1,
    batch_id: 'batch-1',
    session_id: 'session-1',
    queue_id: 'default',
    user_id: 'user-1',
    origin: null,
    destination: null,
    invocation: { id: 'node-1', type: 'l2i' },
    invocation_source_id: 'node-1',
    result: { image: { image_name: imageDTO.image_name } },
    ...overrides,
  }) as unknown as S['InvocationCompleteEvent'];

const dispatchedActionTypes = (dispatch: ReturnType<typeof vi.fn>) =>
  dispatch.mock.calls.map(([action]) => (action as { type?: string }).type);

describe('buildOnInvocationComplete gallery auto-switch', () => {
  const buildHandler = () => {
    const dispatch = vi.fn();
    const getState = vi.fn(() => ({}));
    const handler = buildOnInvocationComplete(getState as never, dispatch as never, new Map());
    return { handler, dispatch };
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getImageDTOSafe).mockResolvedValue(imageDTO);
  });

  // The socket listener routes only the current user's own events here (single-user mode included),
  // so the handler runs unconditionally — ownership is decided upstream via getEventScope.
  it('switches boards and clears the progress indicator when a generation completes', async () => {
    const { handler, dispatch } = buildHandler();

    await handler(buildEvent());

    expect(dispatchedActionTypes(dispatch)).toContain(boardIdSelected.type);
    expect($lastProgressEvent.set).toHaveBeenCalledWith(null);
  });
});

describe('buildOnForeignInvocationComplete', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getImageDTOSafe).mockResolvedValue(imageDTO);
  });

  it('refreshes gallery caches via tag invalidation only', () => {
    const dispatch = vi.fn();
    const handler = buildOnForeignInvocationComplete(dispatch as never);

    handler(buildEvent());

    // Exactly one dispatch: the tag invalidation. No optimistic cache edits, no board/image
    // selection, no progress clear.
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(api.util.invalidateTags).toHaveBeenCalledWith(FOREIGN_GALLERY_REFRESH_TAGS);
    expect(dispatch).toHaveBeenCalledWith({ type: 'api/invalidateTags', payload: FOREIGN_GALLERY_REFRESH_TAGS });
    expect(FOREIGN_GALLERY_REFRESH_TAGS).toEqual(
      expect.arrayContaining(['ImageNameList', 'BoardImagesTotal', 'VirtualBoards'])
    );

    const types = dispatchedActionTypes(dispatch);
    expect(types).not.toContain(boardIdSelected.type);
    expect(types).not.toContain(imageSelected.type);
    expect($lastProgressEvent.set).not.toHaveBeenCalled();
  });

  it('never fetches image DTOs', () => {
    const dispatch = vi.fn();
    const handler = buildOnForeignInvocationComplete(dispatch as never);

    handler(buildEvent());

    expect(getImageDTOSafe).not.toHaveBeenCalled();
  });

  it('does nothing for results without image outputs', () => {
    const dispatch = vi.fn();
    const handler = buildOnForeignInvocationComplete(dispatch as never);

    handler(buildEvent({ result: { latents: { latents_name: 'latents-1' } } }));

    expect(dispatch).not.toHaveBeenCalled();
  });

  it('does nothing for denylisted passthrough node types', () => {
    const dispatch = vi.fn();
    const handler = buildOnForeignInvocationComplete(dispatch as never);

    handler(buildEvent({ invocation: { id: 'node-1', type: 'image' } }));

    expect(dispatch).not.toHaveBeenCalled();
  });
});

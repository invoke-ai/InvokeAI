import { boardIdSelected, imageSelected } from 'features/gallery/store/gallerySlice';
import { boardsApi } from 'services/api/endpoints/boards';
import { getImageDTOSafe } from 'services/api/endpoints/images';
import type { ImageDTO, S } from 'services/api/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { buildOnInvocationComplete } from './onInvocationComplete';

vi.mock('app/logging/logger', () => ({
  logger: () => ({
    debug: vi.fn(),
    trace: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  }),
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
  selectSelectedBoardId: () => 'admin-board',
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

const OWNER_USER_ID = 'user-1';
const OTHER_USER_ID = 'admin-1';

const imageDTO = {
  image_name: 'img-1.png',
  board_id: 'user-board',
  is_intermediate: false,
  image_category: 'general',
} as ImageDTO;

const buildEvent = (): S['InvocationCompleteEvent'] =>
  ({
    item_id: 1,
    batch_id: 'batch-1',
    session_id: 'session-1',
    queue_id: 'default',
    user_id: OWNER_USER_ID,
    origin: null,
    destination: null,
    invocation: { id: 'node-1', type: 'l2i' },
    invocation_source_id: 'node-1',
    result: { image: { image_name: imageDTO.image_name } },
  }) as unknown as S['InvocationCompleteEvent'];

const buildHandler = (currentUser: { user_id: string; is_admin: boolean } | null) => {
  const dispatch = vi.fn();
  const getState = vi.fn(() => ({ auth: { user: currentUser } }));
  const handler = buildOnInvocationComplete(getState as never, dispatch as never, new Map());
  return { handler, dispatch };
};

const dispatchedActionTypes = (dispatch: ReturnType<typeof vi.fn>) =>
  dispatch.mock.calls.map(([action]) => (action as { type?: string }).type);

describe('buildOnInvocationComplete gallery auto-switch', () => {
  beforeEach(() => {
    vi.mocked(getImageDTOSafe).mockResolvedValue(imageDTO);
  });

  it("does not switch boards for another user's generation, but still updates gallery caches", async () => {
    const { handler, dispatch } = buildHandler({ user_id: OTHER_USER_ID, is_admin: true });

    await handler(buildEvent());

    const types = dispatchedActionTypes(dispatch);
    expect(types).not.toContain(boardIdSelected.type);
    expect(types).not.toContain(imageSelected.type);
    // Cache updates still happen so the admin's view of the other user's board stays fresh.
    expect(boardsApi.util.upsertQueryEntries).toHaveBeenCalled();
  });

  it("switches boards for the current user's own generation", async () => {
    const { handler, dispatch } = buildHandler({ user_id: OWNER_USER_ID, is_admin: false });

    await handler(buildEvent());

    expect(dispatchedActionTypes(dispatch)).toContain(boardIdSelected.type);
  });

  it('switches boards in single-user mode (no current user)', async () => {
    const { handler, dispatch } = buildHandler(null);

    await handler(buildEvent());

    expect(dispatchedActionTypes(dispatch)).toContain(boardIdSelected.type);
  });
});

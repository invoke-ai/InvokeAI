import type { GalleryUiAdapter } from '@features/gallery/react';

import { GalleryUiProvider } from '@features/gallery/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, createRef, type Ref, type ReactNode, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryActions } from './GalleryWidgetContext';

import { useGalleryActions } from './useGalleryActions';

const mocks = vi.hoisted(() => ({
  deleteGalleryBoard: vi.fn(),
  invalidateGallery: vi.fn(),
}));

vi.mock('@features/gallery/data/backend', () => ({
  createGalleryBoard: vi.fn(),
  deleteGalleryBoard: (...args: unknown[]) => mocks.deleteGalleryBoard(...args),
  downloadGalleryArchive: vi.fn(),
  updateGalleryBoard: vi.fn(),
  uploadGalleryImage: vi.fn(),
}));

vi.mock('@features/gallery/data/queryCache', () => ({
  invalidateGallery: (...args: unknown[]) => mocks.invalidateGallery(...args),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const actionsRef = createRef<GalleryActions>();
const reconcileDeletedBoardOutcome = vi.fn();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const Probe = ({ ref }: { ref: Ref<GalleryActions> }) => {
  const actions = useGalleryActions({
    boards: [
      {
        archived: false,
        assetCount: 0,
        id: 'board-1',
        imageCount: 2,
        kind: 'board',
        name: 'Board 1',
        videoCount: 1,
      },
    ],
    loadMore: vi.fn(),
    projectBoardId: null,
    projectName: 'Project',
    selectedBoardId: 'board-1',
  });

  useImperativeHandle(ref, () => actions, [actions]);

  return null;
};

const NoopProvider = ({ children }: { children: ReactNode }) => children;
const NoopContextMenu = () => null;
const noop = vi.fn();
const adapter: GalleryUiAdapter = {
  ImageActionsProvider: NoopProvider,
  ImageContextMenu: NoopContextMenu,
  account: { enableLiveFollow: noop },
  antialiasProgressImages: false,
  gallery: {
    reconcileDeletedBoardOutcome,
    selectBoard: noop,
    selectImage: noop,
    setCompareImage: noop,
    setMultiSelection: noop,
    setPage: noop,
    setPageInfo: noop,
    setProjectBoard: noop,
    setSearchTerm: noop,
    setView: noop,
    toggleImageSelection: noop,
    updateSettings: noop,
  },
  galleryValues: {},
  generateValues: {},
  liveFollowEnabled: false,
  liveProgressTarget: null,
  notifications: { add: noop, reportError: noop },
  projectId: 'project-1',
  projectName: 'Project',
  queueItems: [],
  widgets: { patchGalleryValues: noop },
};

beforeEach(async () => {
  vi.clearAllMocks();
  accountLifecycle.activate('user-a');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <GalleryUiProvider adapter={adapter}>
          <Probe ref={actionsRef} />
        </GalleryUiProvider>
      </QueryClientProvider>
    );
  });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('deleteBoard', () => {
  it('forwards the authoritative backend outcome to the Workbench command', async () => {
    const outcome = {
      boardId: 'board-1',
      deletedBoardImageNames: ['retained.png'],
      deletedBoardVideoNames: ['retained.mp4'],
      deletedImageNames: [],
      deletedVideoNames: [],
      failedImageNames: [],
      failedVideoNames: [],
    };
    mocks.deleteGalleryBoard.mockResolvedValue(outcome);

    await act(async () => {
      await actionsRef.current?.deleteBoard('board-1', false);
    });

    expect(reconcileDeletedBoardOutcome).toHaveBeenCalledOnce();
    expect(reconcileDeletedBoardOutcome).toHaveBeenCalledWith(outcome);
  });
});

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
  downloadBlob: vi.fn(),
  downloadGalleryArchive: vi.fn(),
  invalidateGallery: vi.fn(),
  notificationsAdd: vi.fn(),
}));

vi.mock('@features/gallery/data/backend', () => ({
  createGalleryBoard: vi.fn(),
  deleteGalleryBoard: (...args: unknown[]) => mocks.deleteGalleryBoard(...args),
  downloadGalleryArchive: (...args: unknown[]) => mocks.downloadGalleryArchive(...args),
  updateGalleryBoard: vi.fn(),
  uploadGalleryImage: vi.fn(),
}));

vi.mock('@features/gallery/data/queryCache', () => ({
  invalidateGallery: (...args: unknown[]) => mocks.invalidateGallery(...args),
}));

vi.mock('@platform/browser/downloadBlob', () => ({
  downloadBlob: (...args: unknown[]) => mocks.downloadBlob(...args),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const actionsRef = createRef<GalleryActions>();
const reconcileDeletedBoardOutcome = vi.fn();
const setItemMultiSelection = vi.fn();
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
  ItemActionsProvider: NoopProvider,
  ImageContextMenu: NoopContextMenu,
  account: { enableLiveFollow: noop },
  antialiasProgressImages: false,
  gallery: {
    reconcileDeletedBoardOutcome,
    selectBoard: noop,
    selectImage: noop,
    selectItem: noop,
    setCompareImage: noop,
    setCompareItem: noop,
    setItemMultiSelection,
    setMultiSelection: noop,
    setPage: noop,
    setPageInfo: noop,
    setProjectBoard: noop,
    setSearchTerm: noop,
    setView: noop,
    toggleImageSelection: noop,
    toggleItemSelection: noop,
    updateSettings: noop,
  },
  galleryValues: {},
  generateValues: {},
  liveFollowEnabled: false,
  liveProgressTarget: null,
  notifications: { add: (...args: unknown[]) => mocks.notificationsAdd(...args), reportError: noop },
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

describe('mixed item selection', () => {
  it('forwards an ordered same-name mixed range as qualified item keys', () => {
    const primaryItem = {
      boardId: 'board-1',
      category: 'general',
      createdAt: '2026-07-30T00:00:00.000Z',
      durationSeconds: 3,
      fullUrl: '/full/shared',
      height: 64,
      isIntermediate: false,
      kind: 'video',
      name: 'shared',
      starred: false,
      thumbnailUrl: '/thumbnail/shared',
      width: 64,
    } as const;

    actionsRef.current?.selectItemRange(
      [
        { kind: 'image', name: 'shared' },
        { kind: 'video', name: 'shared' },
      ],
      primaryItem
    );

    expect(setItemMultiSelection).toHaveBeenCalledWith(['image:shared', 'video:shared'], primaryItem);
  });
});

describe('board image archive omission', () => {
  it('states the exact existing board video count in the preparation notification without fetching it', async () => {
    mocks.downloadGalleryArchive.mockResolvedValue({
      blob: new Blob(['archive']),
      fileName: 'board-1.zip',
    });

    await act(async () => {
      await actionsRef.current?.downloadBoard('board-1');
    });

    expect(mocks.notificationsAdd).toHaveBeenNthCalledWith(1, {
      kind: 'info',
      message: 'Preparing an image archive of "Board 1". 1 video will be omitted.',
      title: 'Preparing download',
    });
    expect(mocks.downloadGalleryArchive).toHaveBeenCalledWith({
      boardId: 'board-1',
      signal: expect.any(AbortSignal),
    });
  });
});

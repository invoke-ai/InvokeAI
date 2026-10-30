/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryUiAdapter } from '@features/gallery/react';

import { getGallerySettings } from '@features/gallery/core/settings';
import { GalleryUiProvider } from '@features/gallery/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { toaster } from '@platform/ui/toaster';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, createRef, type Ref, type ReactNode, useCallback, useImperativeHandle, useRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryActions } from './GalleryWidgetContext';

import { useGalleryActions } from './useGalleryActions';

const mocks = vi.hoisted(() => ({
  createGalleryBoard: vi.fn(),
  deleteGalleryBoard: vi.fn(),
  ensureProjectOnServer: vi.fn(async () => {}),
  downloadBlob: vi.fn(),
  downloadGalleryArchive: vi.fn(),
  // The ensured listing's answer per board; the panel's `boards` fixture names only board-1 and shared-viewer.
  canAddToCachedGalleryBoard: vi.fn((boardId: string) => boardId === 'board-1'),
  ensureGalleryBoardDestinations: vi.fn((): Promise<void> | null => null),
  getCachedGalleryBoard: vi.fn((_boardId: string): unknown => undefined),
  invalidateGallery: vi.fn(),
  invalidateGalleryBoardDestinations: vi.fn(),
  notificationsAdd: vi.fn(),
  notificationsReportError: vi.fn(),
  notificationsUndoable: vi.fn(),
  projectGalleryValues: vi.fn((_projectId: string): Record<string, unknown> | null => null),
  patchGalleryBoardCaches: vi.fn((..._args: unknown[]) => vi.fn()),
  updateGalleryBoard: vi.fn(),
  uploadGalleryImage: vi.fn(),
  uploadGalleryVideo: vi.fn(),
}));

vi.mock('@features/gallery/data/backend', () => ({
  createGalleryBoard: (...args: unknown[]) => mocks.createGalleryBoard(...args),
  deleteGalleryBoard: (...args: unknown[]) => mocks.deleteGalleryBoard(...args),
  downloadGalleryArchive: (...args: unknown[]) => mocks.downloadGalleryArchive(...args),
  isDateBoardId: (boardId: string) => boardId.startsWith('by_date:'),
  isInvalidGalleryBoardDestination: (boardId: string) => boardId.startsWith('by_date:'),
  updateGalleryBoard: (...args: unknown[]) => mocks.updateGalleryBoard(...args),
  uploadGalleryImage: (...args: unknown[]) => mocks.uploadGalleryImage(...args),
  uploadGalleryVideo: (...args: unknown[]) => mocks.uploadGalleryVideo(...args),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => {
      const messages: Record<string, string> = {
        'widgets.gallery.boardArchivePreparing': `Preparing an image archive of "${String(values?.name)}". ${String(
          values?.count
        )} video will be omitted.`,
        'widgets.gallery.deleteBoardInProgress': `Deleting board "${String(values?.name)}" and its media…`,
        'widgets.gallery.deleteBoardMediaDeleted': `Deleted ${String(values?.media)}.`,
        'widgets.gallery.deleteBoardMediaFailed': `${String(values?.media)} could not be deleted.`,
        'widgets.gallery.deleteBoardMoveOutcome': `Moved ${String(values?.media)} to Uncategorized.`,
        'widgets.gallery.deleteBoardPartialTitle': `Deleted board "${String(values?.name)}" with partial media cleanup`,
        'widgets.gallery.deleteBoardSuccessTitle': `Deleted board "${String(values?.name)}"`,
        'widgets.gallery.downloadReady': 'Download ready',
        'widgets.gallery.imageCount': `${String(values?.count)} ${values?.count === 1 ? 'image' : 'images'}`,
        'widgets.gallery.mediaCountsTwo': `${String(values?.first)} and ${String(values?.second)}`,
        'widgets.gallery.preparingDownload': 'Preparing download',
        'widgets.gallery.uploadDateBoardUnavailable': 'Uploads are unavailable for date boards.',
        'widgets.gallery.uploadFailed': `No files uploaded. ${String(values?.failed)} failed.`,
        'widgets.gallery.uploadPartialTitle': `Uploaded ${String(values?.succeeded)} of ${String(values?.total)} files`,
        'widgets.gallery.uploadFailedCount': `${String(values?.count)} failed.`,
        'widgets.gallery.uploadSplit': 'Images appear in Uploads; videos appear in Media.',
        'widgets.gallery.uploadSummary': `${String(values?.media)} uploaded to ${String(values?.board)}.`,
        'widgets.gallery.uploadSuccessTitle': `Uploaded ${String(values?.count)} files`,
        'widgets.gallery.uploadUnsupported':
          'No supported media files to upload (PNG, JPEG, or WebP images; video files; or audio files).',
        'widgets.gallery.uncategorized': 'Uncategorized',
        'widgets.gallery.videoCount': `${String(values?.count)} ${values?.count === 1 ? 'video' : 'videos'}`,
      };

      return messages[key] ?? key;
    },
  }),
}));

vi.mock('@features/gallery/data/queryCache', () => ({
  addGalleryBoardToCaches: vi.fn(),
  canAddToCachedGalleryBoard: (boardId: string) => mocks.canAddToCachedGalleryBoard(boardId),
  ensureGalleryBoardDestinations: () => mocks.ensureGalleryBoardDestinations(),
  getCachedGalleryBoard: (boardId: string) => mocks.getCachedGalleryBoard(boardId),
  invalidateGallery: (...args: unknown[]) => mocks.invalidateGallery(...args),
  invalidateGalleryBoardDestinations: (...args: unknown[]) => mocks.invalidateGalleryBoardDestinations(...args),
  patchGalleryBoardCaches: (...args: unknown[]) => mocks.patchGalleryBoardCaches(...args),
}));

vi.mock('@platform/browser/downloadBlob', () => ({
  downloadBlob: (...args: unknown[]) => mocks.downloadBlob(...args),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const actionsRef = createRef<GalleryActions>();
const reconcileDeletedBoardOutcome = vi.fn();
const selectBoard = vi.fn();
const updateSettings = vi.fn();
const selectItem = vi.fn();
const setItemMultiSelection = vi.fn();
const patchGalleryValues = vi.fn();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const Probe = ({
  galleryView,
  ref,
  selectedBoardId,
}: {
  galleryView: 'images' | 'assets';
  ref: Ref<GalleryActions>;
  selectedBoardId: string;
}) => {
  const { autoAddBoardId } = getGallerySettings(galleryValues);
  const currentGalleryLocationRef = useRef({ autoAddBoardId, galleryView, selectedBoardId });

  // Match GalleryWidgetView's render-assigned live-read port.
  // eslint-disable-next-line react/refs
  currentGalleryLocationRef.current = { autoAddBoardId, galleryView, selectedBoardId };
  const getCurrentGalleryLocation = useCallback(() => currentGalleryLocationRef.current, []);
  const actions = useGalleryActions({
    boards: [
      {
        archived: false,
        assetCount: 0,
        assetVideoCount: 0,
        id: 'board-1',
        imageCount: 2,
        kind: 'board',
        name: 'Board 1',
        isInbox: false,
        projectId: null,
        videoCount: 1,
      },
      {
        access: { actions: ['read'], adminOverride: false, generalAccess: 'viewer', role: 'viewer' },
        archived: false,
        assetCount: 0,
        assetVideoCount: 0,
        id: 'shared-viewer',
        imageCount: 0,
        isInbox: false,
        kind: 'board',
        name: 'Client references',
        ownerName: 'Cleo',
        projectId: null,
        videoCount: 0,
      },
      {
        archived: false,
        assetCount: 0,
        assetVideoCount: 0,
        id: 'none',
        imageCount: 0,
        kind: 'uncategorized',
        name: '',
        isInbox: false,
        projectId: null,
        videoCount: 0,
      },
    ],
    loadMore: vi.fn(),
    selectedBoardId,
    getCurrentGalleryLocation,
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
  antialiasProgressImages: false,
  exportProject: vi.fn(),
  gallery: {
    clearSelection: noop,
    reconcileDeletedBoardOutcome,
    selectBoard,
    selectImage: noop,
    selectItem,
    setCompareImage: noop,
    setCompareItem: noop,
    setItemMultiSelection,
    setPage: noop,
    setPageInfo: noop,
    setSearchTerm: noop,
    setStarredOnly: noop,
    setSemanticSearchMode: noop,
    setSemanticSearchText: noop,
    commitSemanticSearch: noop,
    clearSearch: noop,
    setView: noop,
    toggleItemSelection: noop,
    updateSettings,
  },
  galleryValues: {},
  generateValues: {},
  getItemLabel: () => Promise.resolve(null),
  getProjectGalleryValues: (projectId: string) => mocks.projectGalleryValues(projectId),
  liveFollowEnabled: false,
  progressSessions: [],
  pinnedProgressSessionId: null,
  followedProgressSessionId: null,
  followProgressSession: vi.fn(),
  notifications: {
    add: (...args: unknown[]) => mocks.notificationsAdd(...args),
    reportError: (...args: unknown[]) => mocks.notificationsReportError(...args),
    undoable: (...args: unknown[]) => mocks.notificationsUndoable(...args),
  },
  projectId: 'project-1',
  projectName: 'Project',
  projects: [],
  ensureProjectOnServer: () => mocks.ensureProjectOnServer(),
  widgets: { openGallery: () => true, patchGalleryValues },
};

/** The shared board as the ensured listing describes it once the caller may only view it. */
const VIEW_ONLY_SHARED_BOARD = {
  access: { actions: ['read'], adminOverride: false, generalAccess: 'viewer', role: 'viewer' },
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id: 'shared-viewer',
  imageCount: 0,
  isInbox: false,
  kind: 'board',
  name: 'Client references',
  ownerName: 'Cleo',
  projectId: null,
  videoCount: 0,
};

let selectedBoardId = 'board-1';
let galleryView: 'images' | 'assets' = 'images';

let galleryValues: Record<string, unknown> = {};

const renderProbe = async () => {
  await act(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <GalleryUiProvider adapter={{ ...adapter, galleryValues }}>
          <Probe galleryView={galleryView} ref={actionsRef} selectedBoardId={selectedBoardId} />
        </GalleryUiProvider>
      </QueryClientProvider>
    );
  });
};

beforeEach(async () => {
  vi.clearAllMocks();
  mocks.projectGalleryValues.mockReset().mockReturnValue(null);
  mocks.canAddToCachedGalleryBoard.mockImplementation((boardId: string) => boardId === 'board-1');
  mocks.ensureGalleryBoardDestinations.mockReturnValue(null);
  mocks.getCachedGalleryBoard.mockReturnValue(undefined);
  accountLifecycle.activate('user-a');
  selectedBoardId = 'board-1';
  galleryView = 'images';
  galleryValues = {};
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await renderProbe();
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
    expect(mocks.notificationsAdd).toHaveBeenCalledWith({
      kind: 'success',
      message: 'Moved 1 image and 1 video to Uncategorized.',
      title: 'Deleted board "Board 1"',
    });
  });

  it('says nothing about media an empty board did not have', async () => {
    mocks.deleteGalleryBoard.mockResolvedValue({
      boardId: 'board-1',
      deletedBoardImageNames: [],
      deletedBoardVideoNames: [],
      deletedImageNames: [],
      deletedVideoNames: [],
      failedImageNames: [],
      failedVideoNames: [],
    });

    await act(async () => {
      await actionsRef.current?.deleteBoard('board-1', false);
    });

    expect(mocks.notificationsAdd).toHaveBeenCalledExactlyOnceWith({
      kind: 'success',
      message: undefined,
      title: 'Deleted board "Board 1"',
    });
  });

  it('shows a media deletion under way until its outcome replaces it, and records only the kinds deleted', async () => {
    const create = vi.spyOn(toaster, 'create');
    const dismiss = vi.spyOn(toaster, 'dismiss');
    let resolveDelete: (outcome: unknown) => void = () => undefined;
    mocks.deleteGalleryBoard.mockReturnValue(
      new Promise((resolve) => {
        resolveDelete = resolve;
      })
    );

    let deletion!: Promise<void> | undefined;
    await act(() => {
      deletion = actionsRef.current?.deleteBoard('board-1', true);
    });

    expect(create).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ title: 'Deleting board "Board 1" and its media…', type: 'loading' })
    );
    expect(dismiss).not.toHaveBeenCalled();

    resolveDelete({
      boardId: 'board-1',
      deletedBoardImageNames: [],
      deletedBoardVideoNames: [],
      deletedImageNames: ['one.png', 'two.png'],
      deletedVideoNames: [],
      failedImageNames: [],
      failedVideoNames: [],
    });
    await act(() => deletion);

    expect(dismiss).toHaveBeenCalledExactlyOnceWith(create.mock.results[0]?.value);
    expect(mocks.notificationsAdd.mock.calls).toEqual([
      [{ kind: 'success', message: 'Deleted 2 images.', title: 'Deleted board "Board 1"' }],
    ]);
    create.mockRestore();
    dismiss.mockRestore();
  });

  it('reports authoritative deleted and failed image/video outcomes', async () => {
    mocks.deleteGalleryBoard.mockResolvedValue({
      boardId: 'board-1',
      deletedBoardImageNames: [],
      deletedBoardVideoNames: [],
      deletedImageNames: ['one.png', 'two.png'],
      deletedVideoNames: ['one.mp4'],
      failedImageNames: ['locked.png'],
      failedVideoNames: ['locked.mp4'],
    });

    await act(async () => {
      await actionsRef.current?.deleteBoard('board-1', true);
    });

    expect(mocks.notificationsAdd).toHaveBeenCalledWith({
      kind: 'success',
      message: 'Deleted 2 images and 1 video. 1 image and 1 video could not be deleted.',
      title: 'Deleted board "Board 1" with partial media cleanup',
    });
  });
});

describe('optimistic board updates', () => {
  it('renames before the request and rolls back when the backend rejects', async () => {
    mocks.updateGalleryBoard.mockRejectedValue(new Error('rename failed'));

    await act(async () => {
      await actionsRef.current?.renameBoard('board-1', 'Renamed');
    });

    expect(mocks.patchGalleryBoardCaches).toHaveBeenCalledWith(expect.anything(), 'board-1', { name: 'Renamed' });
    expect(mocks.patchGalleryBoardCaches.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.updateGalleryBoard.mock.invocationCallOrder[0] ?? 0
    );
    expect(mocks.patchGalleryBoardCaches.mock.results[0]?.value).toHaveBeenCalledOnce();
    expect(mocks.notificationsReportError).toHaveBeenCalledOnce();
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();
  });

  it('moves before the request and rolls back when the backend rejects', async () => {
    mocks.updateGalleryBoard.mockRejectedValue(new Error('move failed'));

    await act(async () => {
      await actionsRef.current?.moveBoard('board-1', null, 'Library');
    });

    expect(mocks.patchGalleryBoardCaches).toHaveBeenCalledWith(expect.anything(), 'board-1', { projectId: null });
    expect(mocks.updateGalleryBoard).toHaveBeenCalledWith('board-1', { projectId: null }, expect.anything());
    expect(mocks.patchGalleryBoardCaches.mock.results[0]?.value).toHaveBeenCalledOnce();
    expect(mocks.notificationsReportError).toHaveBeenCalledOnce();
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();
  });

  it('makes the project exist on the server before creating or moving a board into it, never for the Library', async () => {
    mocks.createGalleryBoard.mockResolvedValue({ id: 'new', name: 'New' });
    mocks.updateGalleryBoard.mockResolvedValue(undefined);
    // Each ensure is held until the test lets it finish, so a request sent without waiting for it shows.
    let finishEnsure: () => void = () => undefined;
    const holdNextEnsure = () =>
      mocks.ensureProjectOnServer.mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            finishEnsure = resolve;
          })
      );

    await act(async () => {
      await actionsRef.current?.createBoard('New', null);
      await actionsRef.current?.moveBoard('board-1', null, 'Library');
    });
    expect(mocks.ensureProjectOnServer).not.toHaveBeenCalled();

    let create!: Promise<boolean | undefined>;
    holdNextEnsure();
    await act(() => {
      create = actionsRef.current!.createBoard('New', 'p1');
    });
    expect(mocks.ensureProjectOnServer).toHaveBeenCalledOnce();
    expect(mocks.createGalleryBoard).toHaveBeenCalledOnce();
    await act(async () => {
      finishEnsure();
      await expect(create).resolves.toBe(true);
    });
    expect(mocks.createGalleryBoard).toHaveBeenLastCalledWith('New', 'p1', expect.anything());

    let move!: Promise<void>;
    holdNextEnsure();
    await act(() => {
      move = actionsRef.current!.moveBoard('board-1', 'p1', 'Project');
    });
    expect(mocks.ensureProjectOnServer).toHaveBeenCalledTimes(2);
    expect(mocks.updateGalleryBoard).toHaveBeenCalledOnce();
    await act(async () => {
      finishEnsure();
      await move;
    });
    expect(mocks.updateGalleryBoard).toHaveBeenLastCalledWith('board-1', { projectId: 'p1' }, expect.anything());
  });

  it('steps the selection and auto-add off a board moved into a project the panel hides, and back on failure', async () => {
    galleryValues = { autoAddBoardId: 'board-1' };
    await renderProbe();
    let rejectUpdate: (error: Error) => void = () => undefined;
    mocks.updateGalleryBoard.mockReturnValue(
      new Promise((_resolve, reject) => {
        rejectUpdate = reject;
      })
    );

    let move!: Promise<void>;
    await act(() => {
      move = actionsRef.current!.moveBoard('board-1', 'other-project', 'Other');
    });

    expect(selectBoard).toHaveBeenCalledExactlyOnceWith('none');
    expect(updateSettings).toHaveBeenCalledExactlyOnceWith({ autoAddBoardId: null });

    // Reflect both optimistic steps in the live gallery the failure path reads.
    selectedBoardId = 'none';
    galleryValues = { autoAddBoardId: null };
    await renderProbe();
    rejectUpdate(new Error('move failed'));
    await act(() => move);

    expect(selectBoard).toHaveBeenLastCalledWith('board-1');
    expect(updateSettings).toHaveBeenLastCalledWith({ autoAddBoardId: 'board-1' });
  });

  it('leaves choices the user made while a failing move was in flight', async () => {
    galleryValues = { autoAddBoardId: 'board-1' };
    await renderProbe();
    let rejectUpdate: (error: Error) => void = () => undefined;
    mocks.updateGalleryBoard.mockReturnValue(
      new Promise((_resolve, reject) => {
        rejectUpdate = reject;
      })
    );

    let move!: Promise<void>;
    await act(() => {
      move = actionsRef.current!.moveBoard('board-1', 'other-project', 'Other');
    });

    // Another board selected, and another made the auto-add target, before the failure arrived.
    selectedBoardId = 'board-2';
    galleryValues = { autoAddBoardId: 'board-3' };
    await renderProbe();
    rejectUpdate(new Error('move failed'));
    await act(() => move);

    expect(selectBoard).toHaveBeenCalledExactlyOnceWith('none');
    expect(updateSettings).toHaveBeenCalledExactlyOnceWith({ autoAddBoardId: null });
  });

  it('keeps the selection and auto-add on a board that stays in sight', async () => {
    mocks.updateGalleryBoard.mockResolvedValue(undefined);

    for (const [values, destination] of [
      [{ autoAddBoardId: 'board-1' }, 'project-1'],
      [{ autoAddBoardId: 'board-1', showOtherProjectBoards: true }, 'other-project'],
    ] as const) {
      galleryValues = values;
      await renderProbe();
      await act(async () => {
        await actionsRef.current?.moveBoard('board-1', destination, 'Somewhere');
      });
    }

    expect(selectBoard).not.toHaveBeenCalled();
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it('keeps a successful rename applied without rolling back', async () => {
    mocks.updateGalleryBoard.mockResolvedValue(undefined);

    await act(async () => {
      await actionsRef.current?.renameBoard('board-1', 'Renamed');
    });

    expect(mocks.patchGalleryBoardCaches.mock.results[0]?.value).not.toHaveBeenCalled();
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
  });

  it('archives the viewed board and steps off it before the request resolves', async () => {
    mocks.updateGalleryBoard.mockResolvedValue(undefined);

    await act(async () => {
      await actionsRef.current?.archiveBoard('board-1', true);
    });

    expect(mocks.patchGalleryBoardCaches).toHaveBeenCalledWith(expect.anything(), 'board-1', { archived: true });
    expect(selectBoard).toHaveBeenCalledWith('none');
    expect(selectBoard.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.updateGalleryBoard.mock.invocationCallOrder[0] ?? 0
    );
  });

  /** Archives the viewed auto-add board and returns its toast's Undo, run until the unarchive request settles. */
  const archiveViewedAutoAddBoard = async () => {
    galleryValues = { autoAddBoardId: 'board-1' };
    await renderProbe();
    mocks.updateGalleryBoard.mockResolvedValue(undefined);
    await act(async () => {
      await actionsRef.current?.archiveBoard('board-1', true);
    });
    const [, { onUndo }] = mocks.notificationsUndoable.mock.calls[0] as [string, { onUndo: () => void }];

    return async () => {
      const requests = mocks.updateGalleryBoard.mock.calls.length;
      await act(async () => {
        onUndo();
        await vi.waitFor(() => expect(mocks.updateGalleryBoard.mock.calls.length).toBe(requests + 1));
        await mocks.updateGalleryBoard.mock.results.at(-1)?.value?.catch(() => undefined);
      });
    };
  };

  it('reports an archive with an Undo that brings the board back and, unchosen since, its results and selection', async () => {
    const undo = await archiveViewedAutoAddBoard();

    expect(mocks.notificationsAdd).not.toHaveBeenCalled();
    expect(mocks.notificationsUndoable).toHaveBeenCalledExactlyOnceWith('widgets.gallery.boardArchived', {
      onUndo: expect.any(Function),
    });
    expect(updateSettings).toHaveBeenCalledExactlyOnceWith({ autoAddBoardId: null });
    expect(selectBoard).toHaveBeenCalledExactlyOnceWith('none');

    // The archiving project, by its id, still sends results to the default and shows Uncategorized.
    mocks.projectGalleryValues.mockImplementation((projectId) =>
      projectId === 'project-1' ? { selectedBoardId: 'none' } : null
    );
    await undo();

    expect(mocks.updateGalleryBoard).toHaveBeenLastCalledWith('board-1', { archived: false }, expect.anything());
    expect(updateSettings).toHaveBeenLastCalledWith({ autoAddBoardId: 'board-1' }, 'project-1');
    expect(selectBoard).toHaveBeenLastCalledWith('board-1');
    // Unarchiving is not itself undone.
    expect(mocks.notificationsUndoable).toHaveBeenCalledOnce();
  });

  it.each([
    ['its project closed', null],
    [
      'its project chose another destination and board since',
      { autoAddBoardId: 'board-3', selectedBoardId: 'board-3' },
    ],
  ])('leaves the destination and selection alone when the archive is undone after %s', async (_case, values) => {
    const undo = await archiveViewedAutoAddBoard();

    mocks.projectGalleryValues.mockReturnValue(values);
    await undo();

    expect(mocks.updateGalleryBoard).toHaveBeenLastCalledWith('board-1', { archived: false }, expect.anything());
    expect(updateSettings).toHaveBeenCalledExactlyOnceWith({ autoAddBoardId: null });
    expect(selectBoard).toHaveBeenCalledExactlyOnceWith('none');
  });

  it('restores nothing when the server refuses to unarchive the board', async () => {
    const undo = await archiveViewedAutoAddBoard();

    mocks.projectGalleryValues.mockReturnValue({ selectedBoardId: 'none' });
    mocks.updateGalleryBoard.mockRejectedValueOnce(new Error('unarchive failed'));
    await undo();

    // Results keep going to the default rather than to a board that is still archived.
    expect(updateSettings).toHaveBeenCalledExactlyOnceWith({ autoAddBoardId: null });
    expect(selectBoard).toHaveBeenCalledExactlyOnceWith('none');
    expect(mocks.notificationsReportError).toHaveBeenCalledOnce();
  });

  it('restores the destination but selects nothing once another project is open', async () => {
    const undo = await archiveViewedAutoAddBoard();

    mocks.projectGalleryValues.mockImplementation((projectId) =>
      projectId === 'project-1' ? { selectedBoardId: 'none' } : null
    );
    await act(() => {
      root?.render(
        <QueryClientProvider client={new QueryClient()}>
          <GalleryUiProvider adapter={{ ...adapter, galleryValues, projectId: 'project-2' }}>
            <Probe galleryView={galleryView} ref={actionsRef} selectedBoardId="none" />
          </GalleryUiProvider>
        </QueryClientProvider>
      );
    });
    await undo();

    expect(updateSettings).toHaveBeenLastCalledWith({ autoAddBoardId: 'board-1' }, 'project-1');
    expect(selectBoard).toHaveBeenCalledExactlyOnceWith('none');
  });

  it('restores the archived board selection on failure only while still on Uncategorized', async () => {
    let rejectUpdate: (error: Error) => void = () => undefined;
    mocks.updateGalleryBoard.mockReturnValue(
      new Promise((_resolve, reject) => {
        rejectUpdate = reject;
      })
    );

    let archive!: Promise<void>;
    await act(() => {
      archive = actionsRef.current!.archiveBoard('board-1', true);
    });

    // Reflect the optimistic step-off in the live location the failure path reads.
    selectedBoardId = 'none';
    await renderProbe();
    rejectUpdate(new Error('archive failed'));
    await act(() => archive);

    expect(mocks.patchGalleryBoardCaches.mock.results[0]?.value).toHaveBeenCalledOnce();
    expect(selectBoard).toHaveBeenNthCalledWith(1, 'none');
    expect(selectBoard).toHaveBeenNthCalledWith(2, 'board-1');
    expect(mocks.notificationsReportError).toHaveBeenCalledOnce();
  });

  it('does not fight a user who navigated elsewhere while an archive was failing', async () => {
    let rejectUpdate: (error: Error) => void = () => undefined;
    mocks.updateGalleryBoard.mockReturnValue(
      new Promise((_resolve, reject) => {
        rejectUpdate = reject;
      })
    );

    let archive!: Promise<void>;
    await act(() => {
      archive = actionsRef.current!.archiveBoard('board-1', true);
    });

    // The user moved on to a different board before the failure arrived.
    selectedBoardId = 'board-2';
    await renderProbe();
    rejectUpdate(new Error('archive failed'));
    await act(() => archive);

    expect(selectBoard).toHaveBeenCalledOnce();
    expect(selectBoard).toHaveBeenCalledWith('none');
  });
});

describe('board access', () => {
  it('sets who can open a board at once, names the setting, and refetches', async () => {
    mocks.updateGalleryBoard.mockResolvedValueOnce({});

    await act(async () => {
      await actionsRef.current?.setBoardGeneralAccess('board-1', 'viewer');
    });

    expect(mocks.patchGalleryBoardCaches).toHaveBeenCalledWith(expect.anything(), 'board-1', {
      generalAccess: 'viewer',
    });
    expect(mocks.updateGalleryBoard).toHaveBeenCalledWith('board-1', { generalAccess: 'viewer' }, expect.anything());
    expect(mocks.notificationsAdd).toHaveBeenCalledWith({
      kind: 'success',
      message: 'Board 1',
      title: 'widgets.gallery.boardAccess.changed',
    });
    expect(mocks.patchGalleryBoardCaches.mock.results[0]?.value).not.toHaveBeenCalled();
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
  });

  it.each([
    // The role no longer allows the action; the board still exists and is readable.
    [
      'board-1',
      403,
      { code: 'board_action_denied', message: 'Not authorized to change this board' },
      'widgets.gallery.boardAccess.stale',
      'Board 1',
    ],
    // The board exists but stopped being readable: an owner restricted it, or a membership was revoked.
    [
      'shared-viewer',
      403,
      { code: 'board_unreadable', message: 'Not authorized to access this board' },
      'widgets.gallery.boardAccess.lost',
      'Client references',
    ],
    // Only a board that is gone answers 404.
    ['board-1', 404, 'Board not found', 'widgets.gallery.boardAccess.gone', 'Board 1'],
  ])(
    'rolls back a refusal of %s answered %i %o, says why, and refetches',
    async (boardId, status, detail, title, name) => {
      mocks.updateGalleryBoard.mockRejectedValueOnce(new ApiError(JSON.stringify({ detail }), status));

      await act(async () => {
        await actionsRef.current?.renameBoard(boardId, 'Renamed');
      });

      expect(mocks.patchGalleryBoardCaches.mock.results[0]?.value).toHaveBeenCalledOnce();
      expect(mocks.notificationsAdd).toHaveBeenCalledWith({ kind: 'info', message: name, title });
      expect(mocks.notificationsReportError).not.toHaveBeenCalled();
      expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
    }
  );

  it.each([
    [
      'set who can open',
      mocks.updateGalleryBoard,
      () => actionsRef.current?.setBoardGeneralAccess('board-1', 'viewer'),
    ],
    ['archive', mocks.updateGalleryBoard, () => actionsRef.current?.archiveBoard('board-1', true)],
    ['delete', mocks.deleteGalleryBoard, () => actionsRef.current?.deleteBoard('board-1', false)],
    ['download', mocks.downloadGalleryArchive, () => actionsRef.current?.downloadBoard('board-1')],
  ])('treats a refused attempt to %s the board as changed access', async (_action, request, run) => {
    request.mockRejectedValueOnce(new ApiError(JSON.stringify({ detail: 'Not allowed' }), 403));

    await act(async () => {
      await run();
    });

    expect(mocks.notificationsAdd).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'info', title: 'widgets.gallery.boardAccess.stale' })
    );
    expect(mocks.notificationsReportError).not.toHaveBeenCalled();
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
  });

  it('reports a move to a project that is gone as the server says, not as lost access', async () => {
    mocks.updateGalleryBoard.mockRejectedValueOnce(new ApiError(JSON.stringify({ detail: 'Project not found' }), 404));

    await act(async () => {
      await actionsRef.current?.moveBoard('board-1', null, 'Library');
    });

    expect(mocks.notificationsReportError).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Project not found' })
    );
    expect(mocks.notificationsAdd).not.toHaveBeenCalled();
    // Either the board or its destination is gone: the listing is refetched all the same.
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
  });
});

describe('archiving the auto-add board', () => {
  const archiveFailingLater = async () => {
    let rejectUpdate: (error: Error) => void = () => undefined;
    mocks.updateGalleryBoard.mockReturnValue(
      new Promise((_resolve, reject) => {
        rejectUpdate = reject;
      })
    );
    let archive!: Promise<void>;
    await act(() => {
      archive = actionsRef.current!.archiveBoard('board-1', true);
    });

    return { archive, reject: () => rejectUpdate(new Error('archive failed')) };
  };

  it('stops auto-adding there, and pins it again if the archive is refused', async () => {
    galleryValues = { autoAddBoardId: 'board-1' };
    await renderProbe();
    const { archive, reject } = await archiveFailingLater();

    expect(updateSettings).toHaveBeenCalledExactlyOnceWith({ autoAddBoardId: null });

    // Reflect the optimistic step in the live gallery the failure path reads.
    galleryValues = { autoAddBoardId: null };
    await renderProbe();
    reject();
    await act(() => archive);

    expect(updateSettings).toHaveBeenLastCalledWith({ autoAddBoardId: 'board-1' });
  });

  it('leaves an auto-add board the user chose while the refused archive was in flight', async () => {
    galleryValues = { autoAddBoardId: 'board-1' };
    await renderProbe();
    const { archive, reject } = await archiveFailingLater();

    galleryValues = { autoAddBoardId: 'board-3' };
    await renderProbe();
    reject();
    await act(() => archive);

    expect(updateSettings).toHaveBeenCalledExactlyOnceWith({ autoAddBoardId: null });
  });
});

describe('localized board labels', () => {
  it('uses the localized Uncategorized label in board notifications', async () => {
    mocks.downloadGalleryArchive.mockResolvedValue({ blob: new Blob(['archive']), fileName: 'gallery.zip' });

    await act(async () => {
      await actionsRef.current?.downloadBoard('none');
    });

    expect(mocks.notificationsAdd).toHaveBeenCalledWith({
      kind: 'info',
      message: 'Preparing an image archive of "Uncategorized". 0 video will be omitted.',
      title: 'Preparing download',
    });
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

const imageUpload = (name: string, queuedAt: string) => ({
  boardId: 'board-1',
  height: 64,
  imageCategory: 'user' as const,
  imageName: name,
  imageUrl: `/images/${name}`,
  queuedAt,
  sourceQueueItemId: 'upload',
  starred: false,
  thumbnailUrl: `/thumbnails/${name}`,
  width: 64,
});

const videoUpload = (name: string, createdAt: string) => ({
  boardId: 'board-1',
  category: 'general' as const,
  createdAt,
  durationSeconds: 4,
  fullUrl: `/videos/${name}`,
  height: 64,
  isIntermediate: false,
  kind: 'video' as const,
  name,
  starred: false,
  thumbnailUrl: `/video-thumbnails/${name}`,
  width: 64,
});

describe('mixed gallery upload', () => {
  it('rejects a date-board upload before starting any request', async () => {
    selectedBoardId = 'by_date:2026-07-30';
    await renderProbe();

    await act(async () => {
      await actionsRef.current?.uploadFiles([new File(['image'], 'photo.png', { type: 'image/png' })]);
    });

    expect(mocks.uploadGalleryImage).not.toHaveBeenCalled();
    expect(mocks.uploadGalleryVideo).not.toHaveBeenCalled();
    expect(mocks.notificationsReportError).toHaveBeenCalledOnce();
    expect(mocks.notificationsReportError).toHaveBeenCalledWith({
      area: 'gallery-upload',
      message: 'Uploads are unavailable for date boards.',
      namespace: 'gallery',
    });
  });

  it('refuses an upload to a board the caller may only view before starting any request', async () => {
    selectedBoardId = 'shared-viewer';
    mocks.getCachedGalleryBoard.mockReturnValue(VIEW_ONLY_SHARED_BOARD);
    await renderProbe();

    await act(async () => {
      await actionsRef.current?.uploadFiles([new File(['image'], 'photo.png', { type: 'image/png' })]);
    });

    expect(mocks.uploadGalleryImage).not.toHaveBeenCalled();
    expect(mocks.notificationsReportError).toHaveBeenCalledExactlyOnceWith({
      area: 'gallery-upload',
      message: 'widgets.gallery.uploadsUnavailableForViewOnlyBoard',
      namespace: 'gallery',
    });
    expect(mocks.invalidateGalleryBoardDestinations).toHaveBeenCalledOnce();
  });

  it.each([
    // The panel still lists board-1 with add_media; the ensured listing holds it as view-only: the role changed.
    ['board-1', 'widgets.gallery.uploadsUnavailableForViewOnlyBoard', { ...VIEW_ONLY_SHARED_BOARD, id: 'board-1' }],
    // Listed by the panel with add_media; the ensured listing no longer holds it: deleted or revoked since.
    ['board-1', 'widgets.gallery.uploadsUnavailableForBoard', undefined],
    // Nothing on screen names it.
    ['gone', 'widgets.gallery.uploadsUnavailableForUnlistedBoard', undefined],
  ])(
    'refuses an upload to %s as the ensured listing says (%s), and reads the listing again',
    async (boardId, message, cachedBoard) => {
      selectedBoardId = boardId;
      mocks.getCachedGalleryBoard.mockReturnValue(cachedBoard);
      await renderProbe();
      let settleListing!: () => void;
      mocks.ensureGalleryBoardDestinations.mockReturnValue(
        new Promise<void>((resolve) => {
          settleListing = resolve;
        })
      );
      mocks.canAddToCachedGalleryBoard.mockReturnValue(false);

      let upload: Promise<unknown> | undefined;
      await act(() => {
        upload = actionsRef.current?.uploadFiles([new File(['image'], 'photo.png', { type: 'image/png' })]);
      });

      // Judged only once the listing is in.
      expect(mocks.ensureGalleryBoardDestinations).toHaveBeenCalledOnce();
      expect(mocks.canAddToCachedGalleryBoard).not.toHaveBeenCalled();
      await act(async () => {
        settleListing();
        await upload;
      });

      expect(mocks.canAddToCachedGalleryBoard).toHaveBeenCalledWith(boardId);
      expect(mocks.uploadGalleryImage).not.toHaveBeenCalled();
      expect(mocks.notificationsReportError).toHaveBeenCalledExactlyOnceWith({
        area: 'gallery-upload',
        message,
        namespace: 'gallery',
      });
      expect(mocks.invalidateGalleryBoardDestinations).toHaveBeenCalledOnce();
      expect(mocks.invalidateGallery).not.toHaveBeenCalled();
    }
  );

  it('sends an upload to Uncategorized without ensuring any listing', async () => {
    selectedBoardId = 'none';
    await renderProbe();
    mocks.uploadGalleryImage.mockResolvedValue(imageUpload('photo.png', '2026-07-30T00:00:00Z'));

    await act(async () => {
      await actionsRef.current?.uploadFiles([new File(['image'], 'photo.png', { type: 'image/png' })]);
    });

    expect(mocks.ensureGalleryBoardDestinations).not.toHaveBeenCalled();
    expect(mocks.uploadGalleryImage).toHaveBeenCalledWith(expect.any(File), 'none', expect.anything());
  });

  it('retains concurrent image uploads while running videos sequentially and continuing after failure', async () => {
    let resolveFirstImage: ((value: ReturnType<typeof imageUpload>) => void) | undefined;
    let resolveSecondImage: ((value: ReturnType<typeof imageUpload>) => void) | undefined;
    let resolveFirstVideo: ((value: ReturnType<typeof videoUpload>) => void) | undefined;
    const firstImage = new Promise<ReturnType<typeof imageUpload>>((resolve) => {
      resolveFirstImage = resolve;
    });
    const secondImage = new Promise<ReturnType<typeof imageUpload>>((resolve) => {
      resolveSecondImage = resolve;
    });
    const firstVideo = new Promise<ReturnType<typeof videoUpload>>((resolve) => {
      resolveFirstVideo = resolve;
    });
    mocks.uploadGalleryImage.mockReturnValueOnce(firstImage).mockReturnValueOnce(secondImage);
    mocks.uploadGalleryVideo
      .mockReturnValueOnce(firstVideo)
      .mockRejectedValueOnce(new Error('bad second video'))
      .mockResolvedValueOnce(videoUpload('third.mp4', '2026-07-30T12:00:05.000Z'));

    let upload: Promise<unknown> | undefined;
    act(() => {
      upload = actionsRef.current?.uploadFiles([
        new File(['image'], 'one.png', { type: 'image/png' }),
        new File(['image'], 'two.webp', { type: 'image/webp' }),
        new File(['video'], 'one.mp4', { type: 'video/mp4' }),
        new File(['video'], 'two.mp4', { type: 'video/mp4' }),
        new File(['video'], 'three.mp4', { type: 'video/mp4' }),
      ]);
    });

    await vi.waitFor(() => {
      expect(mocks.uploadGalleryImage).toHaveBeenCalledTimes(2);
      expect(mocks.uploadGalleryVideo).toHaveBeenCalledTimes(1);
    });

    resolveFirstImage?.(imageUpload('one.png', '2026-07-30T12:00:01.000Z'));
    resolveSecondImage?.(imageUpload('two.webp', '2026-07-30T12:00:02.000Z'));
    resolveFirstVideo?.(videoUpload('one.mp4', '2026-07-30T12:00:03.000Z'));

    await act(async () => {
      await upload;
    });

    expect(mocks.uploadGalleryVideo).toHaveBeenCalledTimes(3);
    expect(mocks.uploadGalleryVideo.mock.invocationCallOrder[1]).toBeLessThan(
      mocks.uploadGalleryVideo.mock.invocationCallOrder[2] ?? Number.POSITIVE_INFINITY
    );
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd).toHaveBeenCalledWith({
      kind: 'success',
      message: '2 images and 2 videos uploaded to Board 1. 1 failed. Images appear in Uploads; videos appear in Media.',
      title: 'Uploaded 4 of 5 files',
    });
  });

  it('selects the newest successful upload visible in the active view', async () => {
    galleryView = 'assets';
    await renderProbe();
    mocks.uploadGalleryImage
      .mockResolvedValueOnce(imageUpload('older.png', '2026-07-30T12:00:01.000Z'))
      .mockResolvedValueOnce(imageUpload('newer.png', '2026-07-30T12:00:04.000Z'));
    mocks.uploadGalleryVideo.mockResolvedValue(videoUpload('newest.mp4', '2026-07-30T12:00:05.000Z'));

    await act(async () => {
      await actionsRef.current?.uploadFiles([
        new File(['image'], 'older.png', { type: 'image/png' }),
        new File(['video'], 'newest.mp4', { type: 'video/mp4' }),
        new File(['image'], 'newer.png', { type: 'image/png' }),
      ]);
    });

    expect(selectItem).toHaveBeenCalledOnce();
    expect(selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'newer.png', category: 'user' })
    );
  });

  it('does not select an upload from the launch board after the active board changes in flight', async () => {
    let resolveUpload: ((value: ReturnType<typeof imageUpload>) => void) | undefined;
    mocks.uploadGalleryImage.mockReturnValueOnce(
      new Promise<ReturnType<typeof imageUpload>>((resolve) => {
        resolveUpload = resolve;
      })
    );

    let upload: Promise<unknown> | undefined;
    act(() => {
      upload = actionsRef.current?.uploadFiles([new File(['image'], 'photo.png', { type: 'image/png' })]);
    });

    await vi.waitFor(() => {
      expect(mocks.uploadGalleryImage).toHaveBeenCalledOnce();
    });
    expect(mocks.uploadGalleryImage).toHaveBeenCalledWith(
      expect.any(File),
      'board-1',
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    );

    selectedBoardId = 'none';
    await renderProbe();
    resolveUpload?.(imageUpload('photo.png', '2026-07-30T12:00:04.000Z'));

    await act(async () => {
      await upload;
    });

    expect(selectItem).not.toHaveBeenCalled();
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd).toHaveBeenCalledOnce();
  });

  it('does not select an upload hidden by a view change while the upload is in flight', async () => {
    galleryView = 'assets';
    await renderProbe();
    let resolveUpload: ((value: ReturnType<typeof imageUpload>) => void) | undefined;
    mocks.uploadGalleryImage.mockReturnValueOnce(
      new Promise<ReturnType<typeof imageUpload>>((resolve) => {
        resolveUpload = resolve;
      })
    );

    let upload: Promise<unknown> | undefined;
    act(() => {
      upload = actionsRef.current?.uploadFiles([new File(['image'], 'photo.png', { type: 'image/png' })]);
    });

    await vi.waitFor(() => {
      expect(mocks.uploadGalleryImage).toHaveBeenCalledOnce();
    });

    galleryView = 'images';
    await renderProbe();
    resolveUpload?.(imageUpload('photo.png', '2026-07-30T12:00:04.000Z'));

    await act(async () => {
      await upload;
    });

    expect(selectItem).not.toHaveBeenCalled();
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd).toHaveBeenCalledOnce();
  });

  it('reports the first actionable rejection when every accepted upload fails alongside unsupported files', async () => {
    mocks.uploadGalleryVideo.mockRejectedValue(new Error('corrupt video'));

    await act(async () => {
      await actionsRef.current?.uploadFiles([
        new File(['text'], 'notes.txt', { type: 'text/plain' }),
        new File(['video'], 'corrupt.mp4', { type: 'video/mp4' }),
      ]);
    });

    expect(mocks.notificationsAdd).not.toHaveBeenCalled();
    expect(mocks.notificationsReportError).toHaveBeenCalledOnce();
    expect(mocks.notificationsReportError).toHaveBeenCalledWith({
      area: 'gallery-upload',
      message: 'corrupt video',
      namespace: 'gallery',
    });
  });

  it('does not schedule another expensive video after the account lifetime aborts', async () => {
    let rejectFirstVideo: ((reason: unknown) => void) | undefined;
    mocks.uploadGalleryVideo.mockReturnValueOnce(
      new Promise((_, reject) => {
        rejectFirstVideo = reject;
      })
    );

    let upload: Promise<unknown> | undefined;
    act(() => {
      upload = actionsRef.current?.uploadFiles([
        new File(['video'], 'one.mp4', { type: 'video/mp4' }),
        new File(['video'], 'two.mp4', { type: 'video/mp4' }),
      ]);
    });

    await vi.waitFor(() => {
      expect(mocks.uploadGalleryVideo).toHaveBeenCalledOnce();
    });
    accountLifecycle.invalidate();
    rejectFirstVideo?.(new DOMException('The operation was aborted.', 'AbortError'));

    await act(async () => {
      await upload;
    });

    expect(mocks.uploadGalleryVideo).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd).not.toHaveBeenCalled();
    expect(mocks.notificationsReportError).not.toHaveBeenCalled();
  });
});

describe('setSemanticImageQuery', () => {
  it('applies the reference while resetting pagination and clearing the text term', () => {
    // Replacing the ranking resets pagination and clears the hidden metadata term.
    actionsRef.current?.setSemanticImageQuery({ imageName: 'ref.png', kind: 'image' });

    expect(patchGalleryValues).toHaveBeenCalledOnce();
    expect(patchGalleryValues).toHaveBeenCalledWith({
      galleryPage: 0,
      searchTerm: '',
      semanticImageQuery: { imageName: 'ref.png', kind: 'image' },
      semanticSearchText: null,
    });
  });

  it('resets pagination when the semantic query is cleared', () => {
    actionsRef.current?.setSemanticImageQuery(null);

    expect(patchGalleryValues).toHaveBeenCalledOnce();
    expect(patchGalleryValues).toHaveBeenCalledWith({
      galleryPage: 0,
      searchTerm: '',
      semanticImageQuery: null,
      semanticSearchText: null,
    });
  });
});

describe('search scope', () => {
  it('starts a newly scoped listing from its first page', () => {
    actionsRef.current?.setSearchScope('all');

    expect(patchGalleryValues).toHaveBeenCalledExactlyOnceWith({ galleryPage: 0, searchScope: 'all' });
  });

  it('refuses uploads while a search looks in every board, before any request', async () => {
    galleryValues = { searchScope: 'all', searchTerm: 'cat' };
    await renderProbe();

    const uploaded = await actionsRef.current?.uploadFiles([new File(['x'], 'a.png', { type: 'image/png' })]);

    expect(uploaded).toEqual([]);
    expect(mocks.ensureGalleryBoardDestinations).not.toHaveBeenCalled();
    expect(mocks.uploadGalleryImage).not.toHaveBeenCalled();
    expect(mocks.notificationsReportError).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'widgets.gallery.uploadsUnavailableForAllBoards' })
    );
  });
});

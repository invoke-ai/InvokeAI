import type { GalleryUiAdapter } from '@features/gallery/react';

import { GalleryUiProvider } from '@features/gallery/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, createRef, type Ref, type ReactNode, useCallback, useImperativeHandle, useRef } from 'react';
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
  notificationsReportError: vi.fn(),
  uploadGalleryImage: vi.fn(),
  uploadGalleryVideo: vi.fn(),
}));

vi.mock('@features/gallery/data/backend', () => ({
  classifyGalleryUpload: (file: File) => {
    const type = file.type.toLowerCase();
    const name = file.name.toLowerCase();

    if (['image/jpeg', 'image/jpg', 'image/png', 'image/webp'].includes(type) || /\.(jpe?g|png|webp)$/.test(name)) {
      return { kind: 'image' as const };
    }
    if (type === 'video/mp4' || name.endsWith('.mp4')) {
      return { kind: 'video' as const };
    }
    return null;
  },
  createGalleryBoard: vi.fn(),
  deleteGalleryBoard: (...args: unknown[]) => mocks.deleteGalleryBoard(...args),
  downloadGalleryArchive: (...args: unknown[]) => mocks.downloadGalleryArchive(...args),
  isDateBoardId: (boardId: string) => boardId.startsWith('by_date:'),
  updateGalleryBoard: vi.fn(),
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
        'widgets.gallery.deleteBoardMediaOutcome': `Deleted ${String(values?.images)} and ${String(
          values?.videos
        )}; ${String(values?.failedImages)} and ${String(values?.failedVideos)} failed.`,
        'widgets.gallery.deleteBoardMoveOutcome': `Moved ${String(values?.images)} and ${String(
          values?.videos
        )} to Uncategorized.`,
        'widgets.gallery.deleteBoardPartialTitle': `Deleted board "${String(values?.name)}" with partial media cleanup`,
        'widgets.gallery.deleteBoardSuccessTitle': `Deleted board "${String(values?.name)}"`,
        'widgets.gallery.downloadReady': 'Download ready',
        'widgets.gallery.imageCount': `${String(values?.count)} images`,
        'widgets.gallery.preparingDownload': 'Preparing download',
        'widgets.gallery.uploadDateBoardUnavailable': 'Uploads are unavailable for date boards.',
        'widgets.gallery.uploadFailed': `No files uploaded. ${String(values?.failed)} failed.`,
        'widgets.gallery.uploadPartialTitle': `Uploaded ${String(values?.succeeded)} of ${String(values?.total)} files`,
        'widgets.gallery.uploadSplit': 'Images appear in Assets; videos appear in Media.',
        'widgets.gallery.uploadSummary': `${String(values?.images)} and ${String(values?.videos)} uploaded to ${String(
          values?.board
        )}. ${String(values?.failed)} failed.`,
        'widgets.gallery.uploadSuccessTitle': `Uploaded ${String(values?.count)} files`,
        'widgets.gallery.uploadUnsupported': 'No supported media files to upload (PNG, JPEG, WebP, or MP4).',
        'widgets.gallery.uncategorized': 'Uncategorized',
        'widgets.gallery.videoCount': `${String(values?.count)} videos`,
      };

      return messages[key] ?? key;
    },
  }),
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
const selectItem = vi.fn();
const setItemMultiSelection = vi.fn();
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
  const currentGalleryLocationRef = useRef({ galleryView, selectedBoardId });

  // Match GalleryWidgetView's render-assigned live-read port.
  // eslint-disable-next-line react/react-compiler
  currentGalleryLocationRef.current = { galleryView, selectedBoardId };
  const getCurrentGalleryLocation = useCallback(() => currentGalleryLocationRef.current, []);
  const actions = useGalleryActions({
    boards: [
      {
        archived: false,
        assetCount: 0,
        id: 'board-1',
        imageCount: 2,
        kind: 'board',
        name: 'Board 1',
        projectId: null,
        videoCount: 1,
      },
      {
        archived: false,
        assetCount: 0,
        id: 'none',
        imageCount: 0,
        kind: 'uncategorized',
        name: '',
        projectId: null,
        videoCount: 0,
      },
    ],
    loadMore: vi.fn(),
    projectBoardId: null,
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
  account: { enableLiveFollow: noop },
  antialiasProgressImages: false,
  exportProject: vi.fn(),
  gallery: {
    reconcileDeletedBoardOutcome,
    selectBoard: noop,
    selectImage: noop,
    selectItem,
    setCompareImage: noop,
    setCompareItem: noop,
    setItemMultiSelection,
    setPage: noop,
    setPageInfo: noop,
    setProjectBoard: noop,
    setSearchTerm: noop,
    setView: noop,
    toggleItemSelection: noop,
    updateSettings: noop,
  },
  galleryValues: {},
  generateValues: {},
  liveFollowEnabled: false,
  liveProgressTarget: null,
  notifications: {
    add: (...args: unknown[]) => mocks.notificationsAdd(...args),
    reportError: (...args: unknown[]) => mocks.notificationsReportError(...args),
  },
  projectId: 'project-1',
  projectName: 'Project',
  queueItems: [],
  widgets: { patchGalleryValues: noop },
};

let selectedBoardId = 'board-1';
let galleryView: 'images' | 'assets' = 'images';

const renderProbe = async () => {
  await act(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <GalleryUiProvider adapter={adapter}>
          <Probe galleryView={galleryView} ref={actionsRef} selectedBoardId={selectedBoardId} />
        </GalleryUiProvider>
      </QueryClientProvider>
    );
  });
};

beforeEach(async () => {
  vi.clearAllMocks();
  accountLifecycle.activate('user-a');
  selectedBoardId = 'board-1';
  galleryView = 'images';
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
      message: 'Moved 1 images and 1 videos to Uncategorized.',
      title: 'Deleted board "Board 1"',
    });
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
      message: 'Deleted 2 images and 1 videos; 1 images and 1 videos failed.',
      title: 'Deleted board "Board 1" with partial media cleanup',
    });
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

    let upload: Promise<void> | undefined;
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
      message: '2 images and 2 videos uploaded to Board 1. 1 failed. Images appear in Assets; videos appear in Media.',
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

    let upload: Promise<void> | undefined;
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

    let upload: Promise<void> | undefined;
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

  it('reports unsupported and rejected files once without aborting supported uploads', async () => {
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
      message: 'No files uploaded. 2 failed.',
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

    let upload: Promise<void> | undefined;
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

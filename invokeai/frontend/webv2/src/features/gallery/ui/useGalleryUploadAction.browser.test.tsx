import type { GalleryUiAdapter } from '@features/gallery/react';

import { GalleryUiProvider } from '@features/gallery/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, createRef, type Ref, type ReactNode, useCallback, useImperativeHandle, useRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryActions } from './GalleryWidgetContext';

import { useGalleryUploadAction } from './useGalleryUploadAction';

const mocks = vi.hoisted(() => ({
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
  isDateBoardId: (boardId: string) => boardId.startsWith('by_date:'),
  uploadGalleryImage: (...args: unknown[]) => mocks.uploadGalleryImage(...args),
  uploadGalleryVideo: (...args: unknown[]) => mocks.uploadGalleryVideo(...args),
}));

vi.mock('@features/gallery/data/queryCache', () => ({
  invalidateGallery: (...args: unknown[]) => mocks.invalidateGallery(...args),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => {
      const messages: Record<string, string> = {
        'widgets.gallery.imageCount': `${String(values?.count)} images`,
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

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const uploadFilesRef = createRef<GalleryActions['uploadFiles']>();
const selectItem = vi.fn();
const noop = vi.fn();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const NoopProvider = ({ children }: { children: ReactNode }) => children;
const adapter: GalleryUiAdapter = {
  ItemActionsProvider: NoopProvider,
  ImageContextMenu: () => null,
  account: { enableLiveFollow: noop },
  antialiasProgressImages: false,
  exportProject: vi.fn(),
  gallery: {
    reconcileDeletedBoardOutcome: noop,
    selectBoard: noop,
    selectImage: noop,
    selectItem,
    setCompareImage: noop,
    setCompareItem: noop,
    setItemMultiSelection: noop,
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

const Probe = ({
  galleryView,
  ref,
  selectedBoardId,
}: {
  galleryView: 'images' | 'assets';
  ref: Ref<GalleryActions['uploadFiles']>;
  selectedBoardId: string;
}) => {
  const currentGalleryLocationRef = useRef({ galleryView, selectedBoardId });

  // eslint-disable-next-line react/react-compiler
  currentGalleryLocationRef.current = { galleryView, selectedBoardId };
  const getCurrentGalleryLocation = useCallback(() => currentGalleryLocationRef.current, []);
  const uploadFiles = useGalleryUploadAction({
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
    getCurrentGalleryLocation,
    selectedBoardId,
  });

  useImperativeHandle(ref, () => uploadFiles, [uploadFiles]);
  return null;
};

let selectedBoardId = 'board-1';
let galleryView: 'images' | 'assets' = 'images';

const renderProbe = async () => {
  await act(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <GalleryUiProvider adapter={adapter}>
          <Probe galleryView={galleryView} ref={uploadFilesRef} selectedBoardId={selectedBoardId} />
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

const imageUpload = (name: string, queuedAt: string, boardId = 'board-1') => ({
  boardId,
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

describe('focused gallery upload action', () => {
  it('rejects a date-board upload before starting any request', async () => {
    selectedBoardId = 'by_date:2026-07-30';
    await renderProbe();

    await act(async () => {
      await uploadFilesRef.current?.([new File(['image'], 'photo.png', { type: 'image/png' })]);
    });

    expect(mocks.uploadGalleryImage).not.toHaveBeenCalled();
    expect(mocks.uploadGalleryVideo).not.toHaveBeenCalled();
    expect(mocks.notificationsReportError).toHaveBeenCalledWith({
      area: 'gallery-upload',
      message: 'Uploads are unavailable for date boards.',
      namespace: 'gallery',
    });
  });

  it('uploads images concurrently while running videos sequentially and continuing after a file failure', async () => {
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
      upload = uploadFilesRef.current?.([
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
    expect(mocks.notificationsAdd).toHaveBeenCalledWith({
      kind: 'success',
      message: '2 images and 2 videos uploaded to Board 1. 1 failed. Images appear in Assets; videos appear in Media.',
      title: 'Uploaded 4 of 5 files',
    });
    expect(selectItem).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ kind: 'video', name: 'third.mp4' }));
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
      upload = uploadFilesRef.current?.([new File(['image'], 'photo.png', { type: 'image/png' })]);
    });
    await vi.waitFor(() => expect(mocks.uploadGalleryImage).toHaveBeenCalledOnce());

    selectedBoardId = 'none';
    await renderProbe();
    resolveUpload?.(imageUpload('photo.png', '2026-07-30T12:00:04.000Z'));

    await act(async () => {
      await upload;
    });

    expect(selectItem).not.toHaveBeenCalled();
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
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
      upload = uploadFilesRef.current?.([
        new File(['video'], 'one.mp4', { type: 'video/mp4' }),
        new File(['video'], 'two.mp4', { type: 'video/mp4' }),
      ]);
    });
    await vi.waitFor(() => expect(mocks.uploadGalleryVideo).toHaveBeenCalledOnce());

    accountLifecycle.invalidate();
    rejectFirstVideo?.(new DOMException('The operation was aborted.', 'AbortError'));
    await act(async () => {
      await upload;
    });

    expect(mocks.uploadGalleryVideo).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd).not.toHaveBeenCalled();
    expect(mocks.notificationsReportError).not.toHaveBeenCalled();
  });

  it('uses the localized Uncategorized label in a successful upload notification', async () => {
    selectedBoardId = 'none';
    await renderProbe();
    mocks.uploadGalleryImage.mockResolvedValue(imageUpload('photo.png', '2026-07-30T12:00:04.000Z', 'none'));

    await act(async () => {
      await uploadFilesRef.current?.([new File(['image'], 'photo.png', { type: 'image/png' })]);
    });

    expect(mocks.notificationsAdd).toHaveBeenCalledWith(
      expect.objectContaining({ message: '1 images and 0 videos uploaded to Uncategorized. 0 failed.' })
    );
  });
});

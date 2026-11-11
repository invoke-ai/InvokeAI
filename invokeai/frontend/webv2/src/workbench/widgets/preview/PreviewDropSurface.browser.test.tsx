import type { GalleryBoard } from '@features/gallery';
import type * as GalleryBackend from '@features/gallery/data/backend';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import {
  getGalleryCompareImage,
  getSelectedGalleryItemFromValues,
  type GalleryImage,
  type GalleryItemKind,
  type GalleryVideoItem,
} from '@features/gallery/contracts';
import { TINY_PNG_BASE64 } from '@platform/browser/nativeDrag.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { commands, page } from 'vitest/browser';

/** The surface drives the real workbench store; the gallery transport is the only double. */
const harness = vi.hoisted(() => ({
  boards: [] as GalleryBoard[],
  store: null as ReturnType<typeof createWorkbenchStore> | null,
  uploadImage: vi.fn<(file: File, boardId: string) => Promise<GalleryImage>>(),
  uploadVideo: vi.fn<(file: File, boardId: string) => Promise<GalleryVideoItem>>(),
}));

vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryBackend>()),
  listGalleryBoards: () => Promise.resolve(harness.boards),
  uploadGalleryImage: (file: File, boardId: string) => harness.uploadImage(file, boardId),
  uploadGalleryVideo: (file: File, boardId: string) => harness.uploadVideo(file, boardId),
}));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useWorkbenchCommands: () => harness.store!.commands,
  useWorkbenchQueries: () => harness.store!.queries,
}));

import { PreviewDropSurface } from './PreviewDropSurface';

const i18n = createInstance();
let root: Root | null = null;
let host: HTMLDivElement | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());
  await i18n.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  Object.assign(harness, { boards: [], store: null });
  harness.uploadImage.mockReset();
  harness.uploadVideo.mockReset();
  root = null;
  host = null;
});

const imageFile = (name: string) =>
  new File([Uint8Array.from(atob(TINY_PNG_BASE64), (char) => char.charCodeAt(0))], name, { type: 'image/png' });
const videoFile = (name: string) => new File(['clip'], name, { type: 'video/mp4' });
const storedImage = (file: { name: string }, createdAt: string, boardId = 'none'): GalleryImage => ({
  boardId,
  createdAt,
  height: 2,
  imageCategory: 'user',
  imageName: `stored-${file.name}`,
  imageUrl: `/images/stored-${file.name}`,
  queuedAt: createdAt,
  sourceQueueItemId: 'upload',
  starred: false,
  thumbnailUrl: `/thumbnails/stored-${file.name}`,
  width: 2,
});
const storedVideo = (file: { name: string }, createdAt: string): GalleryVideoItem => ({
  boardId: 'none',
  category: 'user',
  createdAt,
  durationSeconds: 1,
  fullUrl: `/videos/stored-${file.name}`,
  height: 64,
  isIntermediate: false,
  kind: 'video',
  name: `stored-${file.name}`,
  starred: false,
  thumbnailUrl: `/thumbnails/stored-${file.name}`,
  width: 64,
});
// The first drop loads the upload code; a cold transform can take longer than `waitFor`'s default second.
const LANDED = { timeout: 5_000 };

const render = async ({
  galleryValues,
  isComparing = false,
  isFollowingLive = false,
  shownKind = null,
}: {
  galleryValues?: Record<string, unknown>;
  isComparing?: boolean;
  isFollowingLive?: boolean;
  shownKind?: GalleryItemKind | null;
} = {}) => {
  const store = createWorkbenchStore(createInitialWorkbenchState());
  const projectId = store.getState().activeProjectId;
  if (galleryValues) {
    store.commands.widgets.patchValues('gallery', galleryValues);
  }
  harness.store = store;
  host = document.createElement('div');
  host.style.height = '400px';
  host.style.width = `${Math.min(600, window.innerWidth)}px`;
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={new QueryClient()}>
            <PreviewDropSurface isComparing={isComparing} isFollowingLive={isFollowingLive} shownKind={shownKind}>
              <div data-testid="preview-content" style={{ height: '100%' }} />
            </PreviewDropSurface>
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
  return { projectId, store };
};

const galleryValuesOf = (store: ReturnType<typeof createWorkbenchStore>, projectId: string) =>
  getProjectWidgetValues(
    store.getState().projects.find((project) => project.id === projectId)!,
    'gallery'
  );
const selectedName = (store: ReturnType<typeof createWorkbenchStore>, projectId: string) =>
  getSelectedGalleryItemFromValues(galleryValuesOf(store, projectId))?.name ?? null;
const content = () => page.getByTestId('preview-content').element();
const zone = (name: string) => page.getByText(name, { exact: true });
const zoneTarget = (name: string) => zone(name).element().closest('[data-preview-drop-zone]')!;
const transferOf = (files: File[]) => {
  const transfer = new DataTransfer();
  for (const file of files) {
    transfer.items.add(file);
  }
  return transfer;
};
const send = async (type: string, transfer: DataTransfer, init: DragEventInit = {}, target: Element = content()) => {
  const event = new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: transfer, ...init });
  await act(() => target.dispatchEvent(event));
  return event;
};
const centreOf = (element: Element) => {
  const rect = element.getBoundingClientRect();
  return { clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2 };
};
/** Dispatches a drag over the zone labelled `zoneName` and drops it there, from the test itself. */
const dropOn = async (zoneName: string, files: File[]) => {
  const transfer = transferOf(files);
  await send('dragenter', transfer);
  await expect.element(zone(zoneName)).toBeVisible();
  const at = centreOf(zoneTarget(zoneName));
  await send('dragover', transfer, at);
  return send('drop', transfer, at);
};
/**
 * A real drag from the operating system, through the browser's own drag input, dropped on one half of the surface: the
 * left half shows, the right half compares when comparison is offered.
 */
const nativeDrop = (half: 'left' | 'right', files: { base64: string; name: string }[]) => {
  const rect = content().getBoundingClientRect();
  return commands.nativeDrag({
    files,
    frameWidth: window.innerWidth,
    points: [{ x: rect.left + rect.width * (half === 'left' ? 0.25 : 0.75), y: rect.top + rect.height / 2 }],
  });
};

describe('PreviewDropSurface', () => {
  it('uploads files dropped from the desktop and shows the newest, as an upload in the Gallery does', async () => {
    // Listed after the video, yet the newer of the two: the newest is shown, not the last.
    harness.uploadImage.mockImplementation((file) => Promise.resolve(storedImage(file, '2026-11-10T00:00:02.000Z')));
    harness.uploadVideo.mockImplementation((file) => Promise.resolve(storedVideo(file, '2026-11-10T00:00:01.000Z')));
    const { projectId, store } = await render();

    await nativeDrop('left', [
      { base64: TINY_PNG_BASE64, name: 'still.png' },
      { base64: btoa('clip'), name: 'clip.mp4' },
    ]);

    await vi.waitFor(() => expect(selectedName(store, projectId)).toBe('stored-still.png'), LANDED);
    // To the project's auto-add board (Uncategorized here), each kind by its own route.
    expect(harness.uploadImage.mock.calls.map(([file, boardId]) => [file.name, boardId])).toEqual([
      ['still.png', 'none'],
    ]);
    expect(harness.uploadVideo.mock.calls.map(([file, boardId]) => [file.name, boardId])).toEqual([
      ['clip.mp4', 'none'],
    ]);
    // Selected within its own board, as a pick from Preview's mini gallery is.
    expect(galleryValuesOf(store, projectId).selectedImageQuery).toMatchObject({ boardId: 'none', itemBoard: true });
    expect(store.getState().notifications[0]?.title).toMatch(/^Uploaded 2 /);
    expect(zone('Drop to show').elements()).toHaveLength(0);
  });

  it('compares with a dropped image beside an image shown, as a gallery drag offers', async () => {
    harness.uploadImage.mockImplementation((file) => Promise.resolve(storedImage(file, '2026-11-10T00:00:01.000Z')));
    const { projectId, store } = await render({ shownKind: 'image' });

    await nativeDrop('right', [{ base64: TINY_PNG_BASE64, name: 'other.png' }]);

    await vi.waitFor(
      () => expect(getGalleryCompareImage(galleryValuesOf(store, projectId))?.imageName).toBe('stored-other.png'),
      LANDED
    );
    expect(selectedName(store, projectId)).toBeNull();
  });

  it.each([
    ['live progress is followed', { isFollowingLive: true, shownKind: 'image' as const }],
    ['two images are compared already', { isComparing: true, shownKind: 'image' as const }],
    ['a video is shown', { shownKind: 'video' as const }],
    ['nothing is shown', {}],
  ])('offers no comparison while %s', async (_case, state) => {
    await render(state);

    await send('dragenter', transferOf([imageFile('other.png')]));

    await expect.element(zone('Drop to show')).toBeVisible();
    expect(zone('Drop to compare').elements()).toHaveLength(0);
  });

  it('offers comparison only to a drag that carries an image, reading each new drag afresh', async () => {
    await render({ shownKind: 'image' });

    const videos = transferOf([videoFile('clip.mp4')]);
    await send('dragenter', videos);
    await expect.element(zone('Drop to show')).toBeVisible();
    expect(zone('Drop to compare').elements()).toHaveLength(0);
    await send('dragleave', videos, { relatedTarget: document.body });

    await send('dragenter', transferOf([imageFile('other.png')]));
    await expect.element(zone('Drop to compare')).toBeVisible();
  });

  it('highlights the zone under the files', async () => {
    await render({ shownKind: 'image' });
    const transfer = transferOf([imageFile('other.png')]);

    await send('dragenter', transfer);
    await expect.element(zone('Drop to compare')).toBeVisible();
    await send('dragover', transfer, centreOf(zoneTarget('Drop to compare')));
    expect(zoneTarget('Drop to compare')).toHaveAttribute('data-over');
    expect(zoneTarget('Drop to show')).not.toHaveAttribute('data-over');

    await send('dragover', transfer, centreOf(zoneTarget('Drop to show')));
    expect(zoneTarget('Drop to show')).toHaveAttribute('data-over');
    expect(zoneTarget('Drop to compare')).not.toHaveAttribute('data-over');
  });

  it('uploads to the auto-add board judged from a listing it reads first, not the board the gallery shows', async () => {
    harness.boards = [
      {
        access: { actions: ['add_media'], adminOverride: false, generalAccess: 'restricted', role: 'owner' },
        archived: false,
        assetCount: 0,
        assetVideoCount: 0,
        id: 'board-auto',
        imageCount: 0,
        isInbox: false,
        kind: 'board',
        name: 'Auto',
        projectId: null,
        videoCount: 0,
      },
    ];
    harness.uploadImage.mockImplementation((file, boardId) =>
      Promise.resolve(storedImage(file, '2026-11-10T00:00:01.000Z', boardId))
    );
    const { projectId, store } = await render({
      galleryValues: { autoAddBoardId: 'board-auto', selectedBoardId: 'board-shown' },
    });

    await dropOn('Drop to show', [imageFile('one.png')]);

    await vi.waitFor(() => expect(selectedName(store, projectId)).toBe('stored-one.png'), LANDED);
    expect(harness.uploadImage.mock.calls.map(([, boardId]) => boardId)).toEqual(['board-auto']);
  });

  it('refuses files Preview cannot show with the upload notice, uploading nothing', async () => {
    const { projectId, store } = await render();

    await dropOn('Drop to show', [new File(['notes'], 'notes.txt', { type: 'text/plain' })]);

    await vi.waitFor(
      () =>
        expect(store.getState().notifications[0]).toMatchObject({
          kind: 'error',
          message: 'No supported media files to upload (PNG, JPEG, or WebP images; video files; or audio files).',
        }),
      LANDED
    );
    expect(harness.uploadImage).not.toHaveBeenCalled();
    expect(harness.uploadVideo).not.toHaveBeenCalled();
    expect(selectedName(store, projectId)).toBeNull();
  });

  it('leaves the project switched to alone when an upload lands after the switch', async () => {
    let finishUpload!: () => void;
    harness.uploadImage.mockImplementation(
      (file) =>
        new Promise((resolve) => {
          finishUpload = () => resolve(storedImage(file, '2026-11-10T00:00:01.000Z'));
        })
    );
    const { projectId, store } = await render();

    await dropOn('Drop to show', [imageFile('late.png')]);
    await vi.waitFor(() => expect(harness.uploadImage).toHaveBeenCalledTimes(1), LANDED);
    const switchedTo = await act(() => store.commands.projects.create());
    await act(() => finishUpload());

    await vi.waitFor(() => expect(selectedName(store, projectId)).toBe('stored-late.png'));
    expect(store.getState().activeProjectId).toBe(switchedTo.id);
    expect(selectedName(store, switchedTo.id)).toBeNull();
  });

  it('ends the drag when it leaves Preview or the window, or finishes elsewhere, and ignores drags without files', async () => {
    await render();
    const transfer = transferOf([imageFile('one.png')]);
    const zonesShown = () => zone('Drop to show').elements().length > 0;

    await send('dragenter', transfer);
    expect(zonesShown()).toBe(true);
    await send('dragleave', transfer, { relatedTarget: document.body });
    expect(zonesShown()).toBe(false);

    // Out of the window (or cancelled), after crossing onto an element taken down mid-drag: no leave reports it.
    const doomed = document.createElement('div');
    content().append(doomed);
    await send('dragenter', transfer);
    await send('dragenter', transfer, { relatedTarget: content() }, doomed);
    doomed.remove();
    await send('dragleave', transfer, { relatedTarget: null });
    await vi.waitFor(() => expect(zonesShown()).toBe(false));

    await send('dragenter', transfer);
    await act(() => document.body.dispatchEvent(new DragEvent('drop', { bubbles: true, dataTransfer: transfer })));
    expect(zonesShown()).toBe(false);
    await send('dragenter', transfer);
    await act(() => window.dispatchEvent(new DragEvent('dragend')));
    expect(zonesShown()).toBe(false);

    const text = new DataTransfer();
    text.setData('text/plain', 'words');
    expect((await send('dragenter', text)).defaultPrevented).toBe(false);
    expect((await send('drop', text)).defaultPrevented).toBe(false);
    expect(zonesShown()).toBe(false);
  });
});

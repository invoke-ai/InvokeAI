import type { GalleryImageItem } from '@features/gallery/contracts';
import type { Project } from '@workbench/projectContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { DEFAULT_THEME_ID } from '@theme/themes';
import { isHotkeyModalLayerActive, registerHotkeyModalLayer } from '@workbench/hotkeys/modalLayer';
import { getProjectWidgetInstance } from '@workbench/widgetState';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { PasteMediaRuntime } from './PasteMediaRuntime';

const mocks = vi.hoisted(() => ({
  canUseAsReferenceImage: true,
  project: null as unknown as Project,
  sendToCanvas: vi.fn(),
  uploadFiles: vi.fn(),
  uploadOptions: null as null | { selectedBoardId: string },
  useAsReferenceImage: vi.fn(),
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => mocks.project.id,
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(mocks.project),
  useWidgetValuesSelector: () => ({}),
}));
vi.mock('@features/gallery/react', () => ({
  useGalleryUploadAction: (options: { selectedBoardId: string }) => {
    mocks.uploadOptions = options;
    return mocks.uploadFiles;
  },
}));
vi.mock('@features/gallery/queries', () => ({
  galleryBoardsOptions: () => ({
    queryFn: () =>
      Promise.resolve([
        {
          archived: false,
          assetCount: 0,
          assetVideoCount: 0,
          id: 'board-1',
          imageCount: 0,
          kind: 'board',
          name: 'Moodboard',
        },
      ]),
    queryKey: ['boards'],
  }),
}));
vi.mock('@features/generation/react', () => ({ createGenerateFormValuesSelector: () => () => ({}) }));
vi.mock('@workbench/image-actions', () => ({
  getGalleryCanvasImportMenuItems: () => [
    { destination: 'raster', label: 'widgets.canvas.import.raster', value: 'raster' },
    { destination: 'control', label: 'widgets.canvas.import.control', value: 'control' },
  ],
  useDeletionConfirmation: () => ({ dialog: null, requestDeletionConfirmation: vi.fn() }),
  useImageActions: () => ({
    canUseAsReferenceImage: mocks.canUseAsReferenceImage,
    sendToCanvas: mocks.sendToCanvas,
    useAsReferenceImage: mocks.useAsReferenceImage,
  }),
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
  interpolation: { escapeValue: false },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const uploadedImage: GalleryImageItem = {
  boardId: 'board-1',
  category: 'user',
  createdAt: '2026-09-21T00:00:00.000Z',
  fullUrl: '/full/pasted.png',
  height: 2,
  isIntermediate: false,
  kind: 'image',
  name: 'pasted.png',
  starred: false,
  thumbnailUrl: '/thumb/pasted.png',
  width: 2,
};

// A 2x2 PNG: real bytes so the preview <img> renders instead of logging a decode error.
const PNG_BYTES = Uint8Array.from(
  atob('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAEklEQVR4nGP4z8DwHwyBNAMDAC+tBv1+4dL7AAAAAElFTkSuQmCC'),
  (char) => char.charCodeAt(0)
);
const imageFile = () => new File([PNG_BYTES], 'shot.png', { type: 'image/png' });

const pasteDialog = () => page.getByRole('dialog', { name: /^Pasted file/ });

const paste = (target: EventTarget, files: File[], text?: string): ClipboardEvent => {
  const clipboardData = new DataTransfer();
  files.forEach((file) => clipboardData.items.add(file));
  if (text !== undefined) {
    clipboardData.setData('text/plain', text);
  }
  const event = new ClipboardEvent('paste', { bubbles: true, cancelable: true, clipboardData });
  act(() => target.dispatchEvent(event));
  return event;
};

describe('PasteMediaRuntime', () => {
  let host: HTMLDivElement;
  let root: Root;
  let queryClient: QueryClient;

  beforeEach(async () => {
    mocks.canUseAsReferenceImage = true;
    mocks.project = createInitialWorkbenchState().projects[0]!;
    getProjectWidgetInstance(mocks.project, 'gallery')!.state.values.selectedBoardId = 'board-1';
    applyThemeToRoot(DEFAULT_THEME_ID);
    mocks.sendToCanvas.mockReset().mockResolvedValue(undefined);
    mocks.uploadFiles.mockReset().mockResolvedValue([uploadedImage]);
    mocks.useAsReferenceImage.mockReset();
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <QueryClientProvider client={queryClient}>
              <button type="button">Origin</button>
              <textarea aria-label="Prompt" />
              <div role="dialog" aria-label="Rename">
                <input aria-label="Name" />
              </div>
              <PasteMediaRuntime />
            </QueryClientProvider>
          </I18nextProvider>
        </ChakraProvider>
      )
    );
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('offers destinations for a pasted image and routes the upload to the chosen one', async () => {
    const origin = page.getByRole('button', { name: 'Origin' });
    origin.element().focus();
    const file = imageFile();

    const event = paste(document.body, [file]);

    expect(event.defaultPrevented).toBe(true);
    const dialog = page.getByRole('dialog', { name: i18n.t('shell.pasteMedia.title', { count: 1 }) });
    await expect.element(dialog).toBeVisible();
    await expect.element(page.getByRole('button', { name: /Upload to Gallery/ })).toHaveFocus();
    await expect.element(page.getByText('Moodboard')).toBeVisible();
    await expect
      .element(page.getByRole('img', { name: i18n.t('shell.pasteMedia.imagePreview', { index: 1 }) }))
      .toBeVisible();
    expect(isHotkeyModalLayerActive()).toBe(true);

    await page.getByRole('button', { name: i18n.t('widgets.canvas.import.control') }).click();

    await expect.element(dialog).not.toBeInTheDocument();
    expect(mocks.uploadFiles).toHaveBeenCalledExactlyOnceWith([file]);
    await vi.waitFor(() => expect(mocks.sendToCanvas).toHaveBeenCalledOnce());
    expect(mocks.sendToCanvas.mock.calls[0]?.[0]).toMatchObject([{ imageName: 'pasted.png', width: 2, height: 2 }]);
    expect(mocks.sendToCanvas.mock.calls[0]?.[1]).toBe('control');
    expect(mocks.useAsReferenceImage).not.toHaveBeenCalled();
    await expect.element(origin).toHaveFocus();
    expect(isHotkeyModalLayerActive()).toBe(false);
  });

  it('uploads to the gallery without a follow-on action, and uses images as reference images', async () => {
    paste(document.body, [imageFile()]);
    await page.getByRole('button', { name: /Upload to Gallery/ }).click();
    expect(mocks.uploadFiles).toHaveBeenCalledOnce();
    await vi.waitFor(() => expect(pasteDialog().elements()).toHaveLength(0));
    expect(mocks.sendToCanvas).not.toHaveBeenCalled();

    paste(document.body, [imageFile()]);
    await page.getByRole('button', { name: i18n.t('shell.pasteMedia.reference') }).click();
    await vi.waitFor(() => expect(mocks.useAsReferenceImage).toHaveBeenCalledOnce());
    expect(mocks.useAsReferenceImage.mock.calls[0]?.[0]).toMatchObject({ imageName: 'pasted.png' });
  });

  it('keeps canvas and reference destinations for images only', async () => {
    paste(document.body, [new File(['x'], 'clip.mp4', { type: 'video/mp4' })]);

    await expect.element(pasteDialog()).toBeVisible();
    expect(page.getByRole('button', { name: i18n.t('widgets.canvas.import.raster') }).elements()).toHaveLength(0);
    expect(page.getByRole('button', { name: i18n.t('shell.pasteMedia.reference') }).elements()).toHaveLength(0);
    expect(page.getByText(i18n.t('shell.pasteMedia.videosGalleryOnly')).elements()).toHaveLength(0);
    await expect.element(page.getByRole('button', { name: /Upload to Gallery/ })).toBeVisible();
  });

  it('uploads to the project board while the gallery shows a date bucket', async () => {
    const values = getProjectWidgetInstance(mocks.project, 'gallery')!.state.values;
    values.selectedBoardId = 'by_date:2026-09-21';
    values.projectBoardId = 'board-1';
    paste(document.body, [imageFile()]);

    await expect.element(page.getByRole('button', { name: /Upload to Gallery/ })).toHaveAccessibleName(/Moodboard/);
    expect(mocks.uploadOptions?.selectedBoardId).toBe('board-1');
  });

  it('hides the reference option while the current model cannot take another reference image', async () => {
    mocks.canUseAsReferenceImage = false;
    paste(document.body, [imageFile()]);

    await expect.element(pasteDialog()).toBeVisible();
    await expect.element(page.getByRole('group', { name: i18n.t('shell.pasteMedia.canvasHeading') })).toBeVisible();
    expect(page.getByRole('button', { name: i18n.t('shell.pasteMedia.reference') }).elements()).toHaveLength(0);
  });

  it('sends only the uploaded images of a mixed paste on to the canvas', async () => {
    const video = new File(['x'], 'clip.mp4', { type: 'video/mp4' });
    const uploadedVideo = {
      boardId: 'board-1',
      category: 'user' as const,
      createdAt: '2026-09-21T00:00:00.000Z',
      duration: 1,
      fullUrl: '/full/clip.mp4',
      height: 2,
      isIntermediate: false,
      kind: 'video' as const,
      name: 'clip.mp4',
      starred: false,
      thumbnailUrl: '/thumb/clip.png',
      width: 2,
    };
    mocks.uploadFiles.mockResolvedValue([uploadedVideo, uploadedImage]);
    paste(document.body, [imageFile(), video]);

    await expect
      .element(page.getByRole('dialog', { name: i18n.t('shell.pasteMedia.title', { count: 2 }) }))
      .toBeVisible();
    await expect.element(page.getByText(i18n.t('shell.pasteMedia.videosGalleryOnly'))).toBeVisible();
    await page.getByRole('button', { name: i18n.t('widgets.canvas.import.raster') }).click();

    await vi.waitFor(() => expect(mocks.sendToCanvas).toHaveBeenCalledOnce());
    expect(mocks.sendToCanvas.mock.calls[0]?.[0]).toHaveLength(1);
    expect(mocks.sendToCanvas.mock.calls[0]?.[0]).toMatchObject([{ imageName: 'pasted.png' }]);
  });

  it('leaves text pastes into fields, non-media pastes, and pastes under a modal or inside a dialog alone', async () => {
    const textarea = page.getByRole('textbox', { name: 'Prompt' }).element();
    textarea.focus();
    expect(paste(textarea, [imageFile()], 'caption').defaultPrevented).toBe(false);
    expect(paste(document.body, [], 'just text').defaultPrevented).toBe(false);
    expect(paste(document.body, [new File(['{}'], 'data.json', { type: 'application/json' })]).defaultPrevented).toBe(
      false
    );
    const release = registerHotkeyModalLayer('test-modal');
    expect(paste(document.body, [imageFile()]).defaultPrevented).toBe(false);
    release();
    // Dialogs that register no modal layer still own pastes from inside them.
    const nameInput = page.getByRole('textbox', { name: 'Name' }).element();
    nameInput.focus();
    expect(paste(nameInput, [imageFile()]).defaultPrevented).toBe(false);
    expect(pasteDialog().elements()).toHaveLength(0);

    // A media-only paste into a field is still media.
    expect(paste(textarea, [imageFile()]).defaultPrevented).toBe(true);
    await expect.element(pasteDialog()).toBeVisible();
    await page.getByRole('button', { name: i18n.t('common.cancel') }).click();
    await expect.element(pasteDialog()).not.toBeInTheDocument();
    expect(mocks.uploadFiles).not.toHaveBeenCalled();
  });
});

import type { GenerateReferenceImage } from '@features/generation/core/types';
import type { GenerationUiAdapter } from '@features/generation/ui/GenerationUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { GenerationUiProvider } from '@features/generation/ui/GenerationUiContext';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ReferenceImageCard } from './ReferenceImageCard';

const mocks = vi.hoisted(() => {
  // A real decodable image, so the crop export's <img> loads without a backend.
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const context = canvas.getContext('2d')!;
  context.fillStyle = '#4080c0';
  context.fillRect(0, 0, 512, 512);
  return {
    imageDataUrl: canvas.toDataURL('image/png'),
    upload: vi.fn(),
  };
});

vi.mock('@features/gallery', () => ({ galleryTransfers: { upload: mocks.upload } }));
vi.mock('@features/gallery/utility', () => ({
  galleryImageUrls: { full: () => mocks.imageDataUrl, thumbnail: () => mocks.imageDataUrl },
}));

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({ fallbackLng: 'en', lng: 'en' });

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const noop = vi.fn();
const ADAPTER = {
  gallery: { touchImages: vi.fn() },
  notifications: { info: vi.fn(), reportError: vi.fn() },
} as unknown as GenerationUiAdapter;

const REFERENCE: GenerateReferenceImage = {
  config: {
    image: { original: { image: { height: 512, image_name: 'original.png', width: 512 } } },
    type: 'external_reference_image',
  },
  id: 'reference',
  isEnabled: true,
};

beforeEach(async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={ADAPTER}>
            <ReferenceImageCard
              count={1}
              index={0}
              referenceImage={REFERENCE}
              selectedModel={undefined}
              onFindInGallery={noop}
              onMove={noop}
              onPatch={noop}
              onRemove={noop}
              onUseSize={noop}
            />
          </GenerationUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  vi.clearAllMocks();
});

const openCropDialog = async (): Promise<HTMLElement> => {
  await act(() => userEvent.click(document.querySelector<HTMLElement>('button[aria-label="common.crop"]')!));
  await expect.poll(() => document.querySelector('[role="dialog"]')?.getAttribute('data-state')).toBe('open');
  return document.querySelector<HTMLElement>('[role="dialog"]')!;
};

/** The crop box as percentages of the image, read back from its inline style. */
const readCropBox = (dialog: HTMLElement) => {
  const style = dialog.querySelector<HTMLElement>('[data-crop-box]')!.style;
  return {
    height: parseFloat(style.height),
    width: parseFloat(style.width),
    x: parseFloat(style.left),
    y: parseFloat(style.top),
  };
};

const pointer = (type: string, target: Element, clientX: number, clientY: number) =>
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, buttons: 1, clientX, clientY, isPrimary: true, pointerId: 7 })
  );

describe('reference image crop dialog', () => {
  it('animates out instead of unmounting on close', async () => {
    await act(() => userEvent.click(document.querySelector<HTMLElement>('button[aria-label="common.crop"]')!));
    await expect.poll(() => document.querySelector('[role="dialog"]')?.getAttribute('data-state')).toBe('open');
    const dialog = document.querySelector('[role="dialog"]')!;

    const frames = closingFrames(await recordDialogExit(dialog, () => act(() => userEvent.keyboard('{Escape}'))));
    await expect.poll(() => document.querySelector('[role="dialog"]')).toBeNull();

    expect(frames).not.toHaveLength(0);
  });

  it('holds the chosen aspect ratio while the crop box is resized from a handle', async () => {
    const dialog = await openCropDialog();
    const full = readCropBox(dialog);
    expect(full).toEqual({ height: 100, width: 100, x: 0, y: 0 });

    await act(() => userEvent.click(dialog.querySelector<HTMLElement>('[role="combobox"]')!));
    await act(() => userEvent.click(document.querySelector<HTMLElement>('[role="option"][data-value="9:16"]')!));

    // The source is square, so the box's percentages carry its pixel ratio directly.
    const portrait = readCropBox(dialog);
    expect(portrait.width / portrait.height).toBeCloseTo(9 / 16, 3);
    expect(portrait.height).toBe(100);
    expect(portrait.x + portrait.width / 2).toBeCloseTo(50, 3);

    const stage = dialog.querySelector<HTMLElement>('[data-crop-stage]')!;
    const viewport = stage.parentElement!;
    const handle = dialog.querySelector<HTMLElement>('[data-drag-mode="nw"]')!;
    const rect = stage.getBoundingClientRect();
    const start = handle.getBoundingClientRect();
    const startX = start.left + start.width / 2;
    const startY = start.top + start.height / 2;
    await act(() => {
      pointer('pointerdown', handle, startX, startY);
      pointer('pointermove', viewport, startX + rect.width * 0.1, startY + rect.height * 0.3);
      pointer('pointerup', viewport, startX + rect.width * 0.1, startY + rect.height * 0.3);
    });

    // The pointer shrank the width by 10% and the height by 30%; the ratio follows the larger candidate (the width),
    // as the legacy cropper did, so the box always reaches the pointer. The opposite corner stays anchored.
    const resized = readCropBox(dialog);
    expect(resized.width / resized.height).toBeCloseTo(9 / 16, 3);
    expect(resized.width).toBeCloseTo(portrait.width - 10, 1);
    expect(resized.x + resized.width).toBeCloseTo(portrait.x + portrait.width, 1);
    expect(resized.y + resized.height).toBeCloseTo(100, 1);
    expect(dialog.textContent).toContain(
      `${Math.round((resized.width / 100) * 512)} × ${Math.round((resized.height / 100) * 512)} px`
    );
  });

  it('saves the current crop to Uploads as a durable user image without applying it', async () => {
    mocks.upload.mockResolvedValue({ imageName: 'saved.png' });
    const dialog = await openCropDialog();
    await act(() => userEvent.click(dialog.querySelector<HTMLElement>('[role="combobox"]')!));
    await act(() => userEvent.click(document.querySelector<HTMLElement>('[role="option"][data-value="1:1"]')!));
    const save = [...dialog.querySelectorAll('button')].find(
      (button) => button.textContent === 'widgets.generate.cropSaveToAssets'
    )!;
    await act(() => userEvent.click(save));

    await expect.poll(() => mocks.upload.mock.calls.length).toBe(1);
    const [file, boardId, options] = mocks.upload.mock.calls[0] as [File, string, { isIntermediate: boolean }];
    expect(file).toBeInstanceOf(File);
    expect(file.name).toBe('original-crop.png');
    expect(boardId).toBe('none');
    expect(options.isIntermediate).toBe(false);
    await expect.poll(() => (ADAPTER.notifications.info as ReturnType<typeof vi.fn>).mock.calls.length).toBe(1);
    expect(ADAPTER.gallery.touchImages).toHaveBeenCalledOnce();
    // Nothing was applied: the dialog stays open and the reference untouched.
    expect(dialog.getAttribute('data-state')).toBe('open');
    expect(noop).not.toHaveBeenCalled();
  });
});

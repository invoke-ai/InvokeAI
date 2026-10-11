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
  gallery: { touchImages: vi.fn(), uploadBoardId: 'board-7' },
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

/** A reference whose saved crop is the centred 256px square of the 512px original. */
const SQUARE_CROPPED_REFERENCE: GenerateReferenceImage = {
  ...REFERENCE,
  config: {
    image: {
      crop: {
        box: { height: 256, width: 256, x: 128, y: 128 },
        image: { height: 256, image_name: 'crop.png', width: 256 },
        ratio: null,
      },
      original: { image: { height: 512, image_name: 'original.png', width: 512 } },
    },
    type: 'external_reference_image',
  },
};

const renderCard = (referenceImage: GenerateReferenceImage) =>
  act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={ADAPTER}>
            <ReferenceImageCard
              count={1}
              index={0}
              referenceImage={referenceImage}
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

beforeEach(async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await renderCard(REFERENCE);
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

const pointer = (type: string, target: Element, clientX: number, clientY: number, button = 0) =>
  target.dispatchEvent(
    new PointerEvent(type, {
      bubbles: true,
      button,
      buttons: button === 1 ? 4 : 1,
      clientX,
      clientY,
      isPrimary: true,
      pointerId: 7,
    })
  );

/** `translate(x px, y px) scale(z)` from the stage's inline transform. */
const readView = (dialog: HTMLElement) => {
  const transform = dialog.querySelector<HTMLElement>('[data-crop-stage]')!.style.transform;
  const [, x, y, zoom] = /translate\(([-\d.e]+)px, ([-\d.e]+)px\) scale\(([-\d.e]+)\)/.exec(transform) ?? [];
  return { x: parseFloat(x ?? ''), y: parseFloat(y ?? ''), zoom: parseFloat(zoom ?? '') };
};

const viewportOf = (dialog: HTMLElement) => dialog.querySelector<HTMLElement>('[data-crop-stage]')!.parentElement!;

const spaceKey = (type: 'keydown' | 'keyup', target: EventTarget = window) =>
  target.dispatchEvent(new KeyboardEvent(type, { bubbles: true, cancelable: true, code: 'Space', key: ' ' }));

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

  it('opens the full image free, and a saved crop on the preset it matches', async () => {
    let dialog = await openCropDialog();
    expect(dialog.querySelector('[role="combobox"]')?.textContent).toContain('widgets.generate.cropAspectRatioFree');
    await act(() => userEvent.keyboard('{Escape}'));
    await expect.poll(() => document.querySelector('[role="dialog"]')).toBeNull();

    await renderCard(SQUARE_CROPPED_REFERENCE);
    dialog = await openCropDialog();
    expect(dialog.querySelector('[role="combobox"]')?.textContent).toContain('1:1');
    expect(readCropBox(dialog)).toEqual({ height: 50, width: 50, x: 25, y: 25 });
  });

  it('zooms about the wheel cursor without scrolling the dialog, and pans with Space or the middle button', async () => {
    const dialog = await openCropDialog();
    const viewport = viewportOf(dialog);
    const rect = viewport.getBoundingClientRect();
    expect(readView(dialog)).toEqual({ x: 0, y: 0, zoom: 1 });

    const wheel = new WheelEvent('wheel', {
      bubbles: true,
      cancelable: true,
      clientX: rect.left + 100,
      clientY: rect.top + 50,
      deltaY: -100,
    });
    await act(() => {
      expect(viewport.dispatchEvent(wheel)).toBe(false);
    });
    // The pixel under the cursor stays put: the stage grows by 1.1 about the cursor's viewport offset (the event
    // stores integer client coordinates, so read the offset back from it).
    const zoomed = readView(dialog);
    expect(zoomed.zoom).toBeCloseTo(1.1, 3);
    expect(zoomed.x).toBeCloseTo(-0.1 * (wheel.clientX - rect.left), 3);
    expect(zoomed.y).toBeCloseTo(-0.1 * (wheel.clientY - rect.top), 3);

    // Space arms a pan from anywhere, and a drag on the crop box moves the view rather than the box.
    await act(() => {
      spaceKey('keydown');
    });
    expect(viewport.dataset.panState).toBe('ready');
    const box = dialog.querySelector<HTMLElement>('[data-crop-box]')!;
    const before = readCropBox(dialog);
    await act(() => {
      pointer('pointerdown', box, rect.left + 200, rect.top + 200);
      pointer('pointermove', viewport, rect.left + 230, rect.top + 220);
      pointer('pointerup', viewport, rect.left + 230, rect.top + 220);
    });
    const panned = readView(dialog);
    expect(panned.zoom).toBe(zoomed.zoom);
    expect(panned.x).toBeCloseTo(zoomed.x + 30, 0);
    expect(panned.y).toBeCloseTo(zoomed.y + 20, 0);
    expect(readCropBox(dialog)).toEqual(before);
    // Still held, so the next drag pans too; releasing returns to crop editing.
    expect(viewport.dataset.panState).toBe('ready');
    await act(() => {
      spaceKey('keyup');
    });
    expect(viewport.dataset.panState).toBe('idle');

    await act(() => {
      pointer('pointerdown', viewport, rect.left + 50, rect.top + 50, 1);
      pointer('pointermove', viewport, rect.left + 40, rect.top + 60, 1);
      pointer('pointerup', viewport, rect.left + 40, rect.top + 60, 1);
    });
    const middlePanned = readView(dialog);
    expect(middlePanned.x).toBeCloseTo(panned.x - 10, 0);
    expect(middlePanned.y).toBeCloseTo(panned.y + 10, 0);
    expect(viewport.dataset.panState).toBe('idle');

    await act(() =>
      userEvent.click(
        [...dialog.querySelectorAll('button')].find(
          (b) => b.getAttribute('aria-label') === 'widgets.generate.cropFitView'
        )!
      )
    );
    expect(readView(dialog)).toEqual({ x: 0, y: 0, zoom: 1 });
  });

  it('leaves Space to a focused button, so Apply still activates from the keyboard', async () => {
    const dialog = await openCropDialog();
    const apply = [...dialog.querySelectorAll('button')].find((button) => button.textContent === 'common.apply')!;
    await act(() => apply.focus());

    await act(() => {
      expect(spaceKey('keydown', apply)).toBe(true);
    });
    expect(viewportOf(dialog).dataset.panState).toBe('idle');
    await act(() => userEvent.keyboard(' '));

    // The unchanged full crop applies and closes the dialog.
    await expect.poll(() => noop.mock.calls.length).toBeGreaterThan(0);
    await expect
      .poll(() => document.querySelector('[role="dialog"]')?.getAttribute('data-state') ?? null)
      .not.toBe('open');
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
    expect(boardId).toBe('board-7');
    expect(options.isIntermediate).toBe(false);
    await expect.poll(() => (ADAPTER.notifications.info as ReturnType<typeof vi.fn>).mock.calls.length).toBe(1);
    expect(ADAPTER.gallery.touchImages).toHaveBeenCalledOnce();
    // Nothing was applied: the dialog stays open and the reference untouched.
    expect(dialog.getAttribute('data-state')).toBe('open');
    expect(noop).not.toHaveBeenCalled();
  });
});

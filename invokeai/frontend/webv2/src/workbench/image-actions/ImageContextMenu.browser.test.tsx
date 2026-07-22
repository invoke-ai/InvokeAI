import type { GalleryImage } from '@features/gallery';

/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ImageActions } from './useImageActions';

import { ImageContextMenu, type ImageContextMenuTarget } from './ImageContextMenu';
import { EMPTY_IMAGE_RECALL_CAPABILITIES } from './imageRecall';

const preferences = vi.hoisted(() => ({ confirmImageDeletion: true }));
const NO_BOARDS: [] = [];

vi.mock('@workbench/settings/store', () => ({
  useWorkbenchPreferenceSelector: (selector: (value: { confirmImageDeletion: boolean }) => boolean): boolean =>
    selector(preferences),
}));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => vi.fn() }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({
    generation: { setSource: vi.fn() },
    widgets: { patchValues: vi.fn() },
  }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

const image = (imageName: string): GalleryImage => ({
  boardId: 'none',
  height: 512,
  imageCategory: 'general',
  imageName,
  imageUrl: `/${imageName}`,
  queuedAt: '2026-06-15T00:00:00Z',
  sourceQueueItemId: 'queue-item',
  starred: false,
  thumbnailUrl: `/thumb-${imageName}`,
  width: 512,
});

const createActions = (deleteImages: ImageActions['deleteImages']): ImageActions => ({
  canUseAsReferenceImage: false,
  copyImage: vi.fn(() => Promise.resolve()),
  deleteImages,
  downloadImage: vi.fn(() => Promise.resolve()),
  downloadImages: vi.fn(() => Promise.resolve()),
  getImageRecallCapabilities: vi.fn(() => Promise.resolve(EMPTY_IMAGE_RECALL_CAPABILITIES)),
  moveImagesToBoard: vi.fn(() => Promise.resolve()),
  openImageInPreview: vi.fn(),
  recallImageData: vi.fn(() => Promise.resolve()),
  selectForCompare: vi.fn(),
  sendToCanvas: vi.fn(() => Promise.resolve()),
  setImagesStarred: vi.fn(() => Promise.resolve()),
  useAsReferenceImage: vi.fn(),
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

const getButton = (label: string): HTMLButtonElement => {
  const button = Array.from(document.querySelectorAll<HTMLButtonElement>('button')).find(
    (candidate) => candidate.textContent?.trim() === label
  );

  if (!button) {
    throw new Error(`Could not find button: ${label}`);
  }

  return button;
};

const getMenuItem = (label: string): HTMLElement => {
  const item = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (candidate) => candidate.textContent?.trim() === label
  );

  if (!item) {
    throw new Error(`Could not find menu item: ${label}`);
  }

  return item;
};

const getOpenAlertDialog = (): HTMLElement | null =>
  document.querySelector<HTMLElement>('[role="alertdialog"][data-state="open"]');

const Harness = ({ actions, images }: { actions: ImageActions; images: GalleryImage[] }) => {
  const [target, setTarget] = useState<ImageContextMenuTarget | null>({ images, x: 20, y: 20 });

  return <ImageContextMenu actions={actions} boards={NO_BOARDS} target={target} onClose={() => setTarget(null)} />;
};

const renderMenu = async (actions: ImageActions, images: GalleryImage[]) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await interact(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Harness actions={actions} images={images} />
      </ChakraProvider>
    );
  });
};

beforeEach(() => {
  preferences.confirmImageDeletion = true;
});

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ImageContextMenu deletion confirmation', () => {
  it('keeps single-image confirmation mounted after the host clears the menu target', async () => {
    const deleteImages = vi.fn(() => Promise.resolve());
    await renderMenu(createActions(deleteImages), [image('single.png')]);

    await interact(() => getMenuItem('Delete Image').click());

    expect(getOpenAlertDialog()).not.toBeNull();
    expect(document.body.textContent).toContain('Delete image?');
    expect(deleteImages).not.toHaveBeenCalled();

    await interact(() => getButton('Delete').click());

    expect(deleteImages).toHaveBeenCalledOnce();
    expect(deleteImages).toHaveBeenCalledWith(['single.png']);
    expect(getOpenAlertDialog()).toBeNull();
  });

  it('cancels without deleting', async () => {
    const deleteImages = vi.fn(() => Promise.resolve());
    await renderMenu(createActions(deleteImages), [image('cancel.png')]);

    await interact(() => getMenuItem('Delete Image').click());
    await interact(() => getButton('Cancel').click());

    expect(deleteImages).not.toHaveBeenCalled();
    expect(getOpenAlertDialog()).toBeNull();
  });

  it('deletes immediately when confirmation is disabled', async () => {
    preferences.confirmImageDeletion = false;
    const deleteImages = vi.fn(() => Promise.resolve());
    await renderMenu(createActions(deleteImages), [image('immediate.png')]);

    await interact(() => getMenuItem('Delete Image').click());

    expect(deleteImages).toHaveBeenCalledOnce();
    expect(deleteImages).toHaveBeenCalledWith(['immediate.png']);
    expect(getOpenAlertDialog()).toBeNull();
  });

  it('captures bulk names and cannot be dismissed or submitted twice while deletion is pending', async () => {
    let resolveDeletion: (() => void) | undefined;
    const deletion = new Promise<void>((resolve) => {
      resolveDeletion = resolve;
    });
    const deleteImages = vi.fn(() => deletion);
    await renderMenu(createActions(deleteImages), [image('first.png'), image('second.png')]);

    await interact(() => getMenuItem('Delete Selection').click());

    expect(document.body.textContent).toContain('Delete 2 images?');
    expect(document.body.textContent).toContain('This permanently deletes these images');

    await interact(() => getButton('Delete').click());

    const cancelButton = getButton('Cancel');
    const confirmButton = getButton('Delete');
    const closeButton = document.querySelector<HTMLButtonElement>('button[aria-label="Close"]');
    expect(deleteImages).toHaveBeenCalledOnce();
    expect(deleteImages).toHaveBeenCalledWith(['first.png', 'second.png']);
    expect(cancelButton.disabled).toBe(true);
    expect(confirmButton.disabled).toBe(true);
    expect(closeButton?.disabled).toBe(true);

    await interact(() => {
      confirmButton.click();
      cancelButton.click();
      closeButton?.click();
      document.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Escape' }));
      document.querySelector<HTMLElement>('[data-scope="dialog"][data-part="backdrop"]')?.click();
    });

    expect(deleteImages).toHaveBeenCalledOnce();
    expect(getOpenAlertDialog()).not.toBeNull();

    await interact(() => resolveDeletion?.());

    expect(getOpenAlertDialog()).toBeNull();
  });
});

import type { GalleryImage, GalleryItem, GalleryItemKey } from '@features/gallery';
import type { GalleryItemContextMenuTarget } from '@features/gallery/react';

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

const createActions = (deleteItems: ImageActions['deleteItems']): ImageActions => ({
  canUseAsReferenceImage: false,
  copyImage: vi.fn(() => Promise.resolve()),
  deleteItems,
  deleteImages: vi.fn(() => Promise.resolve()),
  downloadItem: vi.fn(() => Promise.resolve()),
  downloadItems: vi.fn(() => Promise.resolve()),
  downloadImage: vi.fn(() => Promise.resolve()),
  downloadImages: vi.fn(() => Promise.resolve()),
  getImageRecallCapabilities: vi.fn(() => Promise.resolve(EMPTY_IMAGE_RECALL_CAPABILITIES)),
  moveItemsToBoard: vi.fn(() => Promise.resolve()),
  moveImagesToBoard: vi.fn(() => Promise.resolve()),
  openItemInNewTab: vi.fn(),
  openItemInPreview: vi.fn(),
  openImageInPreview: vi.fn(),
  recallImageData: vi.fn(() => Promise.resolve()),
  selectForCompare: vi.fn(),
  sendToCanvas: vi.fn(() => Promise.resolve()),
  setItemsStarred: vi.fn(() => Promise.resolve()),
  setImagesStarred: vi.fn(() => Promise.resolve()),
  savePromptAsTemplate: vi.fn(),
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

const item = (kind: GalleryItem['kind'], name: string): GalleryItem => {
  const base = {
    boardId: 'none',
    category: 'general' as const,
    createdAt: '2026-06-15T00:00:00Z',
    fullUrl: `/full/${name}`,
    height: 512,
    isIntermediate: false,
    name,
    starred: false,
    thumbnailUrl: `/thumb-${name}`,
    width: 512,
  };

  return kind === 'video' ? { ...base, durationSeconds: 12, kind } : { ...base, kind };
};

const renderItemMenu = async (
  actions: ImageActions,
  target: GalleryItemContextMenuTarget,
  previewVideoActions?: {
    isCopyCurrentFrameAvailable: boolean;
    itemKey: GalleryItemKey;
    onCopyCurrentFrame: () => void;
    onOpenDetails: () => void;
  }
) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await interact(() => {
    root?.render(
      <ChakraProvider value={system}>
        <ImageContextMenu
          actions={actions}
          boards={NO_BOARDS}
          previewVideoActions={previewVideoActions}
          target={target as unknown as ImageContextMenuTarget}
          onClose={vi.fn()}
        />
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
    expect(deleteImages).toHaveBeenCalledWith([{ kind: 'image', name: 'single.png' }]);
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
    expect(deleteImages).toHaveBeenCalledWith([{ kind: 'image', name: 'immediate.png' }]);
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
    expect(deleteImages).toHaveBeenCalledWith([
      { kind: 'image', name: 'first.png' },
      { kind: 'image', name: 'second.png' },
    ]);
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

describe('ImageContextMenu mixed-media action visibility', () => {
  it('shows common actions for a single video and hides every image-only action', async () => {
    const video = item('video', 'clip.mp4');
    await renderItemMenu(createActions(vi.fn()), {
      itemRefs: [{ kind: 'video', name: video.name }],
      items: [video],
      x: 20,
      y: 20,
    });

    expect(document.querySelector('[aria-label="Open in new tab"]')).not.toBeNull();
    expect(document.querySelector('[aria-label="Download video"]')).not.toBeNull();
    expect(document.querySelector('[aria-label="Open in preview"]')).not.toBeNull();
    expect(document.querySelector('[aria-label="Star video"]')).not.toBeNull();
    expect(document.body.textContent).toContain('Change Board');
    expect(document.body.textContent).toContain('Delete Video');
    expect(document.body.textContent).not.toContain('Copy to clipboard');
    expect(document.body.textContent).not.toContain('widgets.preview.copyCurrentFrame');
    expect(document.body.textContent).not.toContain('Recall Metadata');
    expect(document.body.textContent).not.toContain('Send to Upscale');
    expect(document.body.textContent).not.toContain('Select for Compare');
    expect(document.body.textContent).not.toContain('New from Image');
  });

  it('shows frame copy and Details only when a Preview host opts a single video into them', async () => {
    const video = item('video', 'preview.mp4');
    const onCopyCurrentFrame = vi.fn();
    const onOpenDetails = vi.fn();
    await renderItemMenu(
      createActions(vi.fn()),
      {
        itemRefs: [{ kind: 'video', name: video.name }],
        items: [video],
        x: 20,
        y: 20,
      },
      { isCopyCurrentFrameAvailable: false, itemKey: 'video:preview.mp4', onCopyCurrentFrame, onOpenDetails }
    );

    const copy = getMenuItem('widgets.preview.copyCurrentFrame');
    const details = getMenuItem('widgets.preview.videoDetails');
    expect(copy.getAttribute('aria-disabled')).toBe('true');

    await interact(() => details.click());
    expect(onOpenDetails).toHaveBeenCalledOnce();
    expect(onCopyCurrentFrame).not.toHaveBeenCalled();
  });

  it('hides Preview video extras when the captured menu target is no longer the selected video', async () => {
    const video = item('video', 'stale-menu.mp4');
    await renderItemMenu(
      createActions(vi.fn()),
      {
        itemRefs: [{ kind: 'video', name: video.name }],
        items: [video],
        x: 20,
        y: 20,
      },
      {
        isCopyCurrentFrameAvailable: true,
        itemKey: 'video:new-selection.mp4',
        onCopyCurrentFrame: vi.fn(),
        onOpenDetails: vi.fn(),
      }
    );

    expect(document.body.textContent).not.toContain('widgets.preview.copyCurrentFrame');
    expect(document.body.textContent).not.toContain('widgets.preview.videoDetails');
  });

  it('opens the primary video from a video-only multi-selection and keeps image-only actions hidden', async () => {
    const primaryVideo = item('video', 'primary.mp4');
    const secondaryVideo = item('video', 'secondary.mp4');
    const actions = createActions(vi.fn());
    await renderItemMenu(actions, {
      itemRefs: [
        { kind: 'video', name: primaryVideo.name },
        { kind: 'video', name: secondaryVideo.name },
      ],
      items: [primaryVideo, secondaryVideo],
      x: 20,
      y: 20,
    });

    const openInNewTab = document.querySelector<HTMLButtonElement>('[aria-label="Open in new tab"]');
    const openInPreview = document.querySelector<HTMLButtonElement>('[aria-label="Open in preview"]');
    expect(openInNewTab).not.toBeNull();
    expect(openInPreview).not.toBeNull();

    await interact(() => {
      openInNewTab?.click();
      openInPreview?.click();
    });

    expect(actions.openItemInNewTab).toHaveBeenCalledWith(primaryVideo);
    expect(actions.openItemInPreview).toHaveBeenCalledWith(primaryVideo);
    expect(document.body.textContent).not.toContain('Copy to clipboard');
    expect(document.body.textContent).not.toContain('Recall Metadata');
    expect(document.body.textContent).not.toContain('Select for Compare');
    expect(document.body.textContent).not.toContain('New from Images');
  });

  it('keeps complete mixed refs for common bulk actions and hides image-only bulk actions when a ref is unresolved', async () => {
    const loadedImage = item('image', 'still.png');
    const actions = createActions(vi.fn());
    const refs = [
      { kind: 'image' as const, name: loadedImage.name },
      { kind: 'video' as const, name: 'unloaded.mp4' },
    ];
    await renderItemMenu(actions, {
      itemRefs: refs,
      items: [loadedImage],
      x: 20,
      y: 20,
    });

    expect(document.body.textContent).toContain('2 items selected');
    expect(document.body.textContent).toContain('Star All');
    expect(document.body.textContent).toContain('Download Selection');
    expect(document.body.textContent).toContain('Change Board');
    expect(document.body.textContent).toContain('Delete Selection');
    expect(document.body.textContent).not.toContain('New from Images');
    const openInNewTab = document.querySelector<HTMLButtonElement>('[aria-label="Open in new tab"]');
    const openInPreview = document.querySelector<HTMLButtonElement>('[aria-label="Open in preview"]');
    expect(openInNewTab).not.toBeNull();
    expect(openInPreview).not.toBeNull();

    await interact(() => {
      openInNewTab?.click();
      openInPreview?.click();
    });
    expect(actions.openItemInNewTab).toHaveBeenCalledWith(loadedImage);
    expect(actions.openItemInPreview).toHaveBeenCalledWith(loadedImage);

    await interact(() => getMenuItem('Star All').click());
    expect(actions.setItemsStarred).toHaveBeenCalledWith(refs, true);
  });
});

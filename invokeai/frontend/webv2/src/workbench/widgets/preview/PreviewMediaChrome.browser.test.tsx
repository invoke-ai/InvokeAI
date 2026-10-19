/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { GalleryImageItem, GalleryVideoItem } from '@features/gallery';
import type { QueueProgressSession } from '@features/queue/contracts';
import type * as queueDevicesModule from '@features/queue/devices';
import type { ImageActions } from '@workbench/image-actions';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { DndContext, PointerSensor, useDndMonitor, useSensor, useSensors, type DragStartEvent } from '@dnd-kit/core';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { PreviewActionStrip } from './PreviewActionStrip';
import { PreviewFilmstrip } from './PreviewFilmstrip';

const sharedImage: GalleryImageItem = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-07-30T12:00:00Z',
  fullUrl: '/images/shared/full',
  height: 720,
  isIntermediate: false,
  kind: 'image',
  name: 'shared',
  starred: false,
  thumbnailUrl: '/images/shared/thumbnail',
  width: 1280,
};

const sharedVideo: GalleryVideoItem = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-07-30T11:00:00Z',
  durationSeconds: 65.1,
  fps: 23.976,
  fullUrl: '/videos/shared/full',
  height: 1080,
  isIntermediate: false,
  kind: 'video',
  name: 'shared',
  starred: true,
  thumbnailUrl: '/videos/shared/thumbnail',
  width: 1920,
};

const mocks = vi.hoisted(() => ({
  itemProgress: null as { device: string | null; percentage: number | null } | null,
  progressImage: null,
  deviceLabel: null as { index: number } | null,
}));

vi.mock('@features/queue/react', () => ({
  consumeQueueItemSwapProgressImage: () => undefined,
  useItemProgress: () => mocks.itemProgress,
  useQueueItemBridgeProgressImage: () => null,
  useQueueItemProgressImage: () => mocks.progressImage,
  useQueueItemSwapProgressImage: () => null,
  useActiveProgressTargets: () => [],
  useActiveProgressTarget: () => null,
  useFollowedProgressTargets: () => [],
  useActiveProgressItemIds: () => [],
  useProgressImage: () => null,
}));

vi.mock('@features/queue/devices', async (importOriginal) => {
  const actual = await importOriginal<typeof queueDevicesModule>();

  return {
    ...actual,
    useDeviceLabel: () => mocks.deviceLabel,
  };
});

vi.mock('@platform/ui/streaming-image/useStreamingImageSource', () => ({
  useStreamingImageSource: () => ({
    alt: 'preview',
    height: 512,
    kind: 'fallback' as const,
    src: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"/>',
    width: 512,
  }),
}));

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: { countOfTotal: '{{count}} of {{total}}', edit: 'Edit', generating: 'Generating' },
        widgets: {
          canvas: { import: { control: 'Control Layer', raster: 'Raster Layer' } },
          gallery: {
            progressPreparing: 'Preparing',
            progressQueued: 'Queued',
            progressSession: '{{name}} {{index}}/{{total}}',
            progressSettling: 'Finishing',
          },
          preview: {
            copyCurrentFrame: 'Copy Current Frame',
            details: 'Details',
            editOnCanvas: 'Edit on Canvas',
            framesPerSecond: '{{count}} fps',
            imageActions: 'Image actions',
            itemCount_one: '{{count}} item',
            itemCount_other: '{{count}} items',
            selectForCompare: 'Select for Compare',
            sendToCanvas: 'Send to Canvas',
            starImage: 'Star image',
            starVideo: 'Star video',
            unstarImage: 'Unstar image',
            unstarVideo: 'Unstar video',
            videoDuration: 'Duration {{duration}}',
          },
          queue: {
            device: {
              shortLabel: 'GPU {{index}}',
            },
          },
        },
      },
    },
  },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let onFilmstripDragStart = vi.fn();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (action: () => void, delay = 0): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, delay);
    });
  });

const render = async (node: ReactNode) => {
  await interact(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>{node}</ChakraProvider>
      </I18nextProvider>
    );
  });
};

const pointer = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, clientX, clientY, isPrimary: true, pointerId: 1 })
  );
};

const FilmstripDragMonitor = () => {
  useDndMonitor({
    onDragStart: (event: DragStartEvent) =>
      onFilmstripDragStart({ data: event.active.data.current, id: event.active.id }),
  });
  return null;
};

const FilmstripDragHarness = () => {
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }));

  return (
    <DndContext sensors={sensors}>
      <FilmstripDragMonitor />
      <PreviewFilmstrip
        density="full"
        items={[sharedImage, sharedVideo]}
        selectedItemKey="image:shared"
        onSelect={() => undefined}
      />
    </DndContext>
  );
};

beforeEach(() => {
  onFilmstripDragStart = vi.fn();
  host = document.createElement('div');
  host.style.cssText = 'height:320px;left:20px;position:fixed;top:20px;width:640px;';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('PreviewFilmstrip mixed media', () => {
  it('uses the gallery-style accent border for the selected item', async () => {
    await render(
      <>
        <Box borderColor="accent.solid" borderWidth="2px" data-filmstrip-selected-border-reference="" />
        <DndContext>
          <PreviewFilmstrip
            density="full"
            items={[sharedImage, sharedVideo]}
            selectedItemKey="image:shared"
            onSelect={() => undefined}
          />
        </DndContext>
      </>
    );

    const reference = host!.querySelector<HTMLElement>('[data-filmstrip-selected-border-reference]')!;
    const selected = host!.querySelector<HTMLButtonElement>('[aria-current="true"]')!;
    const accentColor = getComputedStyle(reference).borderTopColor;

    expect(getComputedStyle(selected).borderTopColor).toBe(accentColor);
    expect(selected.querySelector(':scope > div')).toBeNull();
  });

  it('vertically centers the thumb row inside the fixed-height scroll viewport', async () => {
    await render(
      <DndContext>
        <PreviewFilmstrip
          density="full"
          items={[sharedImage, sharedVideo]}
          selectedItemKey="image:shared"
          onSelect={() => undefined}
        />
      </DndContext>
    );

    const viewport = host!.querySelector<HTMLElement>('[data-scope="scroll-area"][data-part="viewport"]');
    const thumb = host!.querySelector<HTMLButtonElement>('[aria-current="true"]');

    expect(viewport).not.toBeNull();
    expect(thumb).not.toBeNull();

    const viewportRect = viewport!.getBoundingClientRect();
    const thumbRect = thumb!.getBoundingClientRect();
    const viewportCenterY = viewportRect.top + viewportRect.height / 2;
    const thumbCenterY = thumbRect.top + thumbRect.height / 2;

    // Fill the ScrollArea content height so thumbnails center vertically within the strip.
    expect(Math.abs(thumbCenterY - viewportCenterY)).toBeLessThan(1);
  });

  it('keeps same-name media independent and selects the clicked video poster', async () => {
    const onSelect = vi.fn();

    await render(
      <DndContext>
        <PreviewFilmstrip
          density="full"
          items={[sharedImage, sharedVideo]}
          selectedItemKey="video:shared"
          onSelect={onSelect}
        />
      </DndContext>
    );

    const posters = host!.querySelectorAll<HTMLImageElement>('img');
    const imageButton = posters[0]?.closest<HTMLButtonElement>('button');
    const videoButton = posters[1]?.closest<HTMLButtonElement>('button');

    expect(posters).toHaveLength(2);
    expect(posters[0]?.getAttribute('src')).toContain('/images/shared/thumbnail');
    expect(posters[1]?.getAttribute('src')).toContain('/videos/shared/thumbnail');
    expect(imageButton?.getAttribute('aria-current')).toBeNull();
    expect(videoButton?.getAttribute('aria-current')).toBe('true');

    await interact(() => videoButton?.click());

    expect(onSelect).toHaveBeenCalledWith(sharedVideo);
  });

  it('opens the image context menu for a thumb and arms a comparison on alt-click', async () => {
    const onSelect = vi.fn();
    const onCompare = vi.fn();
    const onContextMenu = vi.fn();

    await render(
      <DndContext>
        <PreviewFilmstrip
          density="full"
          items={[sharedImage, sharedVideo]}
          selectedItemKey="video:shared"
          onCompare={onCompare}
          onContextMenu={onContextMenu}
          onSelect={onSelect}
        />
      </DndContext>
    );
    const imageButton = host!.querySelector<HTMLButtonElement>('[aria-label="shared"]')!;

    const contextEvent = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 40, clientY: 50 });
    await interact(() => imageButton.dispatchEvent(contextEvent));
    expect(contextEvent.defaultPrevented).toBe(true);
    expect(onContextMenu).toHaveBeenCalledExactlyOnceWith(sharedImage, 40, 50);

    await interact(() => imageButton.dispatchEvent(new MouseEvent('click', { altKey: true, bubbles: true })));
    expect(onCompare).toHaveBeenCalledExactlyOnceWith(sharedImage);
    expect(onSelect).not.toHaveBeenCalled();

    await interact(() => imageButton.click());
    expect(onSelect).toHaveBeenCalledExactlyOnceWith(sharedImage);
  });

  it('drags a video poster with a qualified filmstrip id and video ref payload', async () => {
    await render(<FilmstripDragHarness />);

    const videoPoster = Array.from(host!.querySelectorAll<HTMLImageElement>('img')).find((image) =>
      image.src.includes('/videos/shared/thumbnail')
    );
    const videoButton = videoPoster?.closest<HTMLButtonElement>('button');

    if (!videoButton) {
      throw new Error('Expected the video poster button.');
    }

    await interact(() => pointer('pointerdown', videoButton, 120, 80), 20);
    await interact(() => pointer('pointermove', videoButton.ownerDocument, 150, 80), 50);

    expect(onFilmstripDragStart).toHaveBeenCalledWith({
      data: { items: [{ kind: 'video', name: 'shared' }], kind: 'gallery-item' },
      id: 'preview-filmstrip:video:shared',
    });

    await interact(() => pointer('pointerup', videoButton.ownerDocument, 150, 80), 300);
  });

  it('never falls back to the protected full video URL when a poster is unavailable', async () => {
    const videoWithoutPoster = {
      ...sharedVideo,
      fullUrl: '/protected/videos/shared.mp4',
      thumbnailUrl: '',
    };

    await render(
      <DndContext>
        <PreviewFilmstrip
          density="full"
          items={[sharedImage, videoWithoutPoster]}
          selectedItemKey="video:shared"
          onSelect={() => undefined}
        />
      </DndContext>
    );

    const videoButton = host?.querySelector<HTMLButtonElement>('[aria-label="Video shared"]');

    expect(videoButton?.querySelector('img')).toBeNull();
    expect(
      [...host!.querySelectorAll<HTMLImageElement>('img')].map((image) => image.getAttribute('src'))
    ).not.toContain(videoWithoutPoster.fullUrl);
  });
});

describe('Preview mixed media actions', () => {
  it('keeps common video actions, adds the Preview-only frame copy, and hides image-only actions', async () => {
    const actions = {
      downloadItem: vi.fn(() => Promise.resolve()),
      setItemsStarred: vi.fn(() => Promise.resolve()),
    } as unknown as ImageActions;
    const onCopyCurrentFrame = vi.fn();

    await render(
      <PreviewActionStrip
        actions={actions}
        density="full"
        isVideoFrameCopyAvailable={false}
        item={sharedVideo}
        onCopyCurrentFrame={onCopyCurrentFrame}
        onOpenMenu={() => undefined}
      />
    );

    // Image-only verbs: a video has no compare partner and no canvas
    // destination.
    expect(host?.querySelector('[aria-label="Select for Compare"]')).toBeNull();
    expect(host?.querySelector('[aria-label="Edit on Canvas"]')).toBeNull();

    const copyFrame = host?.querySelector<HTMLButtonElement>('[aria-label="Copy Current Frame"]');
    const star = host?.querySelector<HTMLButtonElement>('[aria-label="Unstar video"]');
    expect(copyFrame).not.toBeNull();
    expect(copyFrame?.disabled).toBe(true);
    // Details lives beside the header toggles now, not in the strip.
    expect(host?.querySelector('[aria-label="Details"]')).toBeNull();
    expect(star).not.toBeNull();

    await interact(() => star?.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true })));

    expect(onCopyCurrentFrame).not.toHaveBeenCalled();
    expect(actions.setItemsStarred).toHaveBeenCalledWith([{ kind: 'video', name: 'shared' }], false);

    await render(
      <PreviewActionStrip
        actions={actions}
        density="full"
        isVideoFrameCopyAvailable
        item={sharedVideo}
        onCopyCurrentFrame={onCopyCurrentFrame}
        onOpenMenu={() => undefined}
      />
    );
    const enabledCopyFrame = host?.querySelector<HTMLButtonElement>('[aria-label="Copy Current Frame"]');
    expect(enabledCopyFrame?.disabled).toBe(false);

    await interact(() => enabledCopyFrame?.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true })));
    expect(onCopyCurrentFrame).toHaveBeenCalledOnce();
  });

  it('opens Edit onto the canvas layer destinations and sends the image to the chosen one', async () => {
    const actions = {
      copyImage: vi.fn(() => Promise.resolve()),
      downloadItem: vi.fn(() => Promise.resolve()),
      selectForCompare: vi.fn(),
      sendToCanvas: vi.fn(() => Promise.resolve()),
      setItemsStarred: vi.fn(() => Promise.resolve()),
    } as unknown as ImageActions;

    await render(
      <PreviewActionStrip actions={actions} density="full" item={sharedImage} onOpenMenu={() => undefined} />
    );
    const editOnCanvas = host?.querySelector<HTMLButtonElement>('[aria-label="Edit on Canvas"]');

    expect(editOnCanvas).not.toBeNull();
    // Leads the strip and carries a visible label, not just a glyph.
    expect(host?.querySelector('button')).toBe(editOnCanvas);
    expect(editOnCanvas?.textContent).toContain('Edit');
    expect(host?.querySelector('[aria-label="Copy Current Frame"]')).toBeNull();

    await page.getByRole('button', { name: 'Edit on Canvas' }).click();
    await expect.element(page.getByRole('menuitem', { name: 'Raster Layer' })).toBeVisible();
    await page.getByRole('menuitem', { name: 'Control Layer' }).click();
    expect(actions.sendToCanvas).toHaveBeenCalledWith([expect.objectContaining({ imageName: 'shared' })], 'control');
  });

  it('leaves copy and download to the image-actions dropdown', async () => {
    const actions = {
      copyImage: vi.fn(() => Promise.resolve()),
      downloadItem: vi.fn(() => Promise.resolve()),
      selectForCompare: vi.fn(),
      sendToCanvas: vi.fn(() => Promise.resolve()),
      setItemsStarred: vi.fn(() => Promise.resolve()),
    } as unknown as ImageActions;

    await render(
      <PreviewActionStrip actions={actions} density="full" item={sharedImage} onOpenMenu={() => undefined} />
    );

    expect(host?.querySelector('[aria-label="Copy to clipboard"]')).toBeNull();
    expect(host?.querySelector('[aria-label="Download image"]')).toBeNull();
    expect(host?.querySelector('[aria-label="Image actions"]')).not.toBeNull();
  });
});

describe('PreviewFilmstrip live sessions', () => {
  const session = (id: string, state: QueueProgressSession['state'], backendItemId = 1): QueueProgressSession => ({
    backendItemId,
    height: 512,
    id,
    itemCount: 2,
    itemIndex: Number(id.split(':')[1]),
    label: 'Generate',
    queueItemId: 'queue-1',
    sourceId: 'generate',
    state,
    width: 512,
  });

  it('leads the strip with one thumb per slot, naming device and progress, and follows a running one on click', async () => {
    mocks.itemProgress = { device: 'cuda:1', percentage: 0.4 };
    mocks.deviceLabel = { index: 1 };
    const onFollowSession = vi.fn();
    const onUnpinSession = vi.fn();

    // A single board item would hide the strip; the live sessions earn it.
    await render(
      <DndContext>
        <PreviewFilmstrip
          density="full"
          followedSessionId="queue-1:1"
          items={[sharedImage]}
          selectedItemKey={null}
          sessions={[session('queue-1:1', 'running'), session('queue-1:2', 'queued', 2)]}
          onFollowSession={onFollowSession}
          onSelect={() => undefined}
          onUnpinSession={onUnpinSession}
        />
      </DndContext>
    );

    const thumbs = [...host!.querySelectorAll<HTMLButtonElement>('[data-preview-live-thumb]')];
    expect(thumbs.map((thumb) => thumb.getAttribute('aria-label'))).toEqual([
      'Generate 1/2 · GPU 1 · 40%',
      'Generate 2/2 · GPU 1 · Queued',
    ]);
    // Live thumbs come first; the item thumb follows them.
    expect(host!.querySelector('button')).toBe(thumbs[0]);
    expect(thumbs[0]?.getAttribute('aria-current')).toBe('true');
    expect(thumbs[1]?.getAttribute('aria-current')).toBeNull();
    expect(thumbs[1]?.getAttribute('aria-disabled')).toBe('true');

    await interact(() => thumbs[1]?.click());
    expect(onFollowSession).not.toHaveBeenCalled();
    await interact(() => thumbs[0]?.click());
    expect(onFollowSession).toHaveBeenCalledExactlyOnceWith('queue-1:1');
    expect(onUnpinSession).not.toHaveBeenCalled();
  });

  it('reads as a pressed toggle while pinned and unpins on the next click', async () => {
    mocks.itemProgress = { device: null, percentage: null };
    mocks.deviceLabel = null;
    const onFollowSession = vi.fn();
    const onUnpinSession = vi.fn();

    await render(
      <DndContext>
        <PreviewFilmstrip
          density="compact"
          followedSessionId="queue-1:1"
          isSessionPinned
          items={[]}
          selectedItemKey={null}
          sessions={[session('queue-1:1', 'running'), session('queue-1:2', 'settling', 2)]}
          onFollowSession={onFollowSession}
          onSelect={() => undefined}
          onUnpinSession={onUnpinSession}
        />
      </DndContext>
    );

    const [pinned, settling] = [...host!.querySelectorAll<HTMLButtonElement>('[data-preview-live-thumb]')];
    // A silent thumb reads as a stuck one, so it says it is preparing.
    expect(pinned?.getAttribute('aria-label')).toBe('Generate 1/2 · Preparing');
    expect(pinned?.getAttribute('aria-pressed')).toBe('true');
    // The pin shows in place of the progress ring, so the next click reads as "unpin".
    expect(pinned?.hasAttribute('data-preview-live-pinned')).toBe(true);
    expect(pinned?.querySelector('[data-scope="progress-circle"]')).toBeNull();
    expect(settling?.hasAttribute('data-preview-live-pinned')).toBe(false);
    expect(settling?.getAttribute('aria-label')).toBe('Generate 2/2 · Finishing');
    expect(settling?.getAttribute('aria-pressed')).toBeNull();

    await interact(() => pinned?.click());
    expect(onUnpinSession).toHaveBeenCalledOnce();
    expect(onFollowSession).not.toHaveBeenCalled();
  });

  it('earns the strip for a lone item only while a session is live', async () => {
    mocks.itemProgress = null;
    mocks.deviceLabel = null;
    const strip = () => host!.querySelector('[data-preview-filmstrip]');

    await render(
      <DndContext>
        <PreviewFilmstrip density="full" items={[sharedImage]} selectedItemKey={null} onSelect={() => undefined} />
      </DndContext>
    );
    expect(strip()).toBeNull();

    await render(
      <DndContext>
        <PreviewFilmstrip
          density="full"
          items={[sharedImage]}
          selectedItemKey={null}
          sessions={[session('queue-1:1', 'running')]}
          onSelect={() => undefined}
        />
      </DndContext>
    );
    expect(strip()).not.toBeNull();
  });

  it('hands focus back to the preview region when the focused live thumb finishes', async () => {
    mocks.itemProgress = { device: null, percentage: 0.5 };
    mocks.deviceLabel = null;
    const strip = (sessions: QueueProgressSession[]) => (
      // The preview's navigation boundary is a focusable region too.
      // oxlint-disable-next-line jsx-a11y/no-noninteractive-tabindex
      <div role="region" tabIndex={0} data-testid="preview-region">
        <DndContext>
          <PreviewFilmstrip
            density="full"
            items={[sharedImage, sharedVideo]}
            selectedItemKey={null}
            sessions={sessions}
            onSelect={() => undefined}
          />
        </DndContext>
      </div>
    );

    await render(strip([session('queue-1:1', 'running')]));
    const thumb = host!.querySelector<HTMLButtonElement>('[data-preview-live-thumb]')!;
    await interact(() => thumb.focus());
    expect(document.activeElement).toBe(thumb);

    await render(strip([]));
    await interact(() => undefined);
    expect(document.activeElement).toBe(host!.querySelector('[data-testid="preview-region"]'));
  });

  it('keeps focus on a thumb that becomes the followed one', async () => {
    mocks.itemProgress = { device: null, percentage: 0.5 };
    mocks.deviceLabel = null;
    const strip = (followedSessionId: string | null) => (
      // oxlint-disable-next-line jsx-a11y/no-noninteractive-tabindex
      <div role="region" tabIndex={0}>
        <DndContext>
          <PreviewFilmstrip
            density="full"
            followedSessionId={followedSessionId}
            items={[]}
            selectedItemKey={null}
            sessions={[session('queue-1:1', 'running'), session('queue-1:2', 'running', 2)]}
            onSelect={() => undefined}
          />
        </DndContext>
      </div>
    );

    await render(strip('queue-1:1'));
    const second = host!.querySelector<HTMLButtonElement>('[data-preview-live-thumb="queue-1:2"]')!;
    await interact(() => second.focus());

    // The click that follows it re-renders it as current; focus must stay put.
    await render(strip('queue-1:2'));
    await interact(() => undefined);
    expect(document.activeElement).toBe(second);
    expect(second.getAttribute('aria-current')).toBe('true');
  });
});

/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { GalleryImageItem, GalleryVideoItem } from '@features/gallery';
import type { ImageActions } from '@workbench/image-actions';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext, PointerSensor, useDndMonitor, useSensor, useSensors, type DragStartEvent } from '@dnd-kit/core';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { PreviewActionStrip } from './PreviewActionStrip';
import { PreviewFilmstrip } from './PreviewFilmstrip';
import { PreviewFooter } from './PreviewFooter';

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

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: { countOfTotal: '{{count}} of {{total}}' },
        widgets: {
          preview: {
            framesPerSecond: '{{count}} fps',
            itemCount_one: '{{count}} item',
            itemCount_other: '{{count}} items',
            nextItemInBoard: 'Next item in board',
            previousItemInBoard: 'Previous item in board',
            videoDuration: 'Duration {{duration}}',
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
});

describe('Preview mixed media footer and actions', () => {
  it('shows mixed position, dimensions, duration, and localized fps with tabular numerals', async () => {
    await render(
      <PreviewFooter
        actionImage={null}
        actions={{} as ImageActions}
        boardItemCount={3}
        isLoadingBoard={false}
        isMetadataOpen={false}
        item={sharedVideo}
        selectedIndex={1}
        onNext={() => undefined}
        onPrevious={() => undefined}
        onToggleMetadata={() => undefined}
      />
    );

    expect(host?.textContent).toContain('2 of 3');
    expect(host?.textContent).toContain('1920 × 1080');
    expect(host?.textContent).toContain('Duration 1:06');
    expect(host?.textContent).toContain('23.976 fps');

    const status = Array.from(host!.querySelectorAll<HTMLElement>('p')).find((element) =>
      element.textContent?.includes('1920 × 1080')
    );
    expect(status ? getComputedStyle(status).fontVariantNumeric : '').toContain('tabular-nums');
    expect(host?.querySelector('[aria-label="Previous item in board"]')).not.toBeNull();
    expect(host?.querySelector('[aria-label="Next item in board"]')).not.toBeNull();
  });

  it('keeps common video actions and hides image-only compare and clipboard actions', async () => {
    const actions = {
      downloadItem: vi.fn(() => Promise.resolve()),
      setItemsStarred: vi.fn(() => Promise.resolve()),
    } as unknown as ImageActions;

    await render(
      <PreviewActionStrip actions={actions} density="full" item={sharedVideo} onOpenMenu={() => undefined} />
    );

    expect(host?.querySelector('[aria-label="Select for Compare"]')).toBeNull();
    expect(host?.querySelector('[aria-label="Copy to clipboard"]')).toBeNull();

    const star = host?.querySelector<HTMLButtonElement>('[aria-label="Unstar video"]');
    const download = host?.querySelector<HTMLButtonElement>('[aria-label="Download video"]');
    expect(star).not.toBeNull();
    expect(download).not.toBeNull();

    await interact(() => star?.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true })));
    await interact(() => download?.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true })));

    expect(actions.setItemsStarred).toHaveBeenCalledWith([{ kind: 'video', name: 'shared' }], false);
    expect(actions.downloadItem).toHaveBeenCalledWith(sharedVideo);
  });
});

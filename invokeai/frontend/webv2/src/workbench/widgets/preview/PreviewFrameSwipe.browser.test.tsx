/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { GalleryItem } from '@features/gallery';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext, useDndMonitor, useDroppable, useSensor, useSensors } from '@dnd-kit/core';
import { system } from '@theme/system';
import { HoldToDragSensor, PrimaryMouseSensor } from '@workbench/shell/holdToDragSensor';
import { widgetCollisionDetection } from '@workbench/widgetDnd';
import { createInstance } from 'i18next';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { PreviewNeighbor } from './usePreviewNavigation';

import { PreviewFrame } from './PreviewFrame';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: { widgets: { preview: { dragVideo: 'Drag video' } } } } },
});

// The production hold, so a slow runner cannot arm a drag between a touch and its first quick move.
const HOLD_DELAY_MS = 400;
const STAGE_WIDTH = 300;

const svg = (fill: string) =>
  `data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128"><rect width="128" height="128" fill="${fill}"/></svg>`;

const makeItem = (name: string, kind: 'image' | 'video' = 'image'): GalleryItem => {
  const base = {
    boardId: 'none',
    category: 'general' as const,
    createdAt: '2026-01-01T00:00:00Z',
    fullUrl: svg(name),
    height: 128,
    isIntermediate: false,
    name,
    starred: false,
    thumbnailUrl: svg(name),
    width: 128,
  };

  return kind === 'image' ? { ...base, kind } : { ...base, durationSeconds: 1, kind };
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (action: () => void, ms = 50): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, ms);
    });
  });

/**
 * Dispatch a touch pointer event stamped at `at` ms on a virtual gesture clock: release velocity is read from event
 * timestamps, so stamping them keeps flicks and slow drags independent of how fast the runner is.
 */
const touch = (
  type: string,
  target: EventTarget,
  clientX: number,
  clientY: number,
  at: number,
  init: PointerEventInit = {}
) => {
  const event = new PointerEvent(type, {
    bubbles: true,
    button: type === 'pointermove' ? -1 : 0,
    cancelable: true,
    clientX,
    clientY,
    isPrimary: true,
    pointerId: 1,
    pointerType: 'touch',
    ...init,
  });

  Object.defineProperty(event, 'timeStamp', { value: at });
  target.dispatchEvent(event);
};

/** Horizontal touch path: down at the first x, a move per later x, then up, each `stepMs` apart on the gesture clock. */
const swipe = async (
  target: HTMLElement,
  xs: number[],
  { stepMs = 16, y = 140 }: { stepMs?: number; y?: number } = {}
) => {
  const base = performance.now();

  await interact(() => touch('pointerdown', target, xs[0]!, y, base), 0);

  for (const [index, x] of xs.slice(1).entries()) {
    await interact(() => touch('pointermove', document, x, y, base + (index + 1) * stepMs), 0);
  }

  await interact(() => touch('pointerup', document, xs.at(-1)!, y, base + xs.length * stepMs), 0);
};

const isAtRest = (element: HTMLElement): boolean => ['', 'none'].includes(element.style.translate);

const DropTarget = () => {
  const { setNodeRef } = useDroppable({ data: { kind: 'test-drop' }, id: 'test-drop' });

  return <div ref={setNodeRef} style={{ height: 120, left: 420, position: 'fixed', top: 40, width: 120 }} />;
};

const DropMonitor = ({ onDrop }: { onDrop: (overId: string | number | null) => void }) => {
  useDndMonitor({ onDragEnd: (event) => onDrop(event.over?.id ?? null) });

  return null;
};

/**
 * A three-item carousel whose navigation moves a real selection, so a committed swipe re-renders the frame with the
 * next source the way Preview does.
 */
const renderCarousel = async ({
  items = [makeItem('red'), makeItem('green'), makeItem('blue')],
  navigateResult = true,
  next,
  start = 1,
}: {
  items?: GalleryItem[];
  /** `pending` never settles, like a page fetch that hangs. */
  navigateResult?: boolean | 'pending';
  /** Overrides the next neighbor derived from `items`. */
  next?: PreviewNeighbor;
  start?: number;
} = {}) => {
  const onNavigate = vi.fn<(direction: -1 | 1) => Promise<boolean>>();
  const onDrop = vi.fn();

  const Carousel = () => {
    const [index, setIndex] = useState(start);
    const item = items[index]!;
    const toNeighbor = (candidate: GalleryItem | undefined): PreviewNeighbor =>
      candidate ? { item: candidate, kind: 'item' } : null;

    onNavigate.mockImplementation((direction) => {
      if (navigateResult === 'pending') {
        return new Promise<boolean>(() => {});
      }

      if (navigateResult) {
        setIndex((current) => current + direction);
      }

      return Promise.resolve(navigateResult);
    });

    return (
      <PreviewFrame
        dragItem={{ kind: item.kind, name: item.name }}
        frameHeight={item.height}
        frameWidth={item.width}
        isLive={false}
        shouldAntialiasLiveImage
        source={
          item.kind === 'image'
            ? {
                itemKey: `image:${item.name}`,
                kind: 'image',
                source: { alt: item.name, height: 128, kind: 'fallback', src: item.fullUrl, width: 128 },
              }
            : {
                itemKey: `video:${item.name}`,
                kind: 'video',
                label: item.name,
                poster: item.thumbnailUrl,
                src: item.fullUrl,
              }
        }
        swipe={{
          neighbors: {
            next: next === undefined ? toNeighbor(items[index + 1]) : next,
            previous: toNeighbor(items[index - 1]),
          },
          onNavigate,
        }}
        variant="framed"
      />
    );
  };

  const Harness = () => {
    const sensors = useSensors(
      useSensor(PrimaryMouseSensor, { activationConstraint: { distance: 6 } }),
      useSensor(HoldToDragSensor, { activationConstraint: { delay: HOLD_DELAY_MS, tolerance: 10 } })
    );

    return (
      <DndContext collisionDetection={widgetCollisionDetection} sensors={sensors}>
        <DropMonitor onDrop={onDrop} />
        <div style={{ display: 'flex', height: 240, left: 40, position: 'fixed', top: 40, width: STAGE_WIDTH }}>
          <Carousel />
        </div>
        <DropTarget />
      </DndContext>
    );
  };

  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await interact(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  const media = () => host!.querySelector<HTMLElement>('img[alt]:not([alt=""]), video')!;

  return {
    content: () => media().parentElement!,
    media,
    neighbors: () => [...host!.querySelectorAll<HTMLElement>('[data-swipe-neighbor]')],
    onDrop,
    onNavigate,
  };
};

const offsetOf = (element: HTMLElement): number => Number.parseFloat(element.style.translate) || 0;

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('PreviewFrame touch swipe', () => {
  it('flicks to the next image and lands the new selection at rest in the same commit', async () => {
    const carousel = await renderCarousel();
    const image = carousel.media();

    expect(image.getAttribute('alt')).toBe('green');

    // Record where the track sits in the very task that mounts the new image, before anything can paint.
    const frame = carousel.content();
    const offsetsAtSwap: string[] = [];
    const observer = new MutationObserver(() => offsetsAtSwap.push(frame.style.translate));

    observer.observe(frame, { childList: true });

    const base = performance.now();

    await interact(() => touch('pointerdown', image, 200, 140, base));
    await interact(() => touch('pointermove', document, 170, 140, base + 16));
    await interact(() => touch('pointermove', document, 130, 140, base + 32));

    // Mid-gesture the image and its neighbors follow the finger.
    expect(offsetOf(carousel.content())).toBe(-70);
    expect(carousel.neighbors().map(offsetOf)).toEqual([-70, -70]);

    await interact(() => touch('pointerup', document, 130, 140, base + 48));
    await vi.waitFor(() => expect(carousel.media().getAttribute('alt')).toBe('blue'));
    observer.disconnect();

    expect(carousel.onNavigate).toHaveBeenCalledExactlyOnceWith(1);
    // The parked offset never reaches a painted frame of the new image: it and both neighbors are back at rest.
    expect(offsetsAtSwap.length).toBeGreaterThan(0);
    expect(offsetsAtSwap.every((offset) => ['', 'none'].includes(offset))).toBe(true);
    // The new image is a new element: a reused one would keep painting the old picture until the new one loaded.
    expect(carousel.media()).not.toBe(image);
    expect(isAtRest(carousel.content())).toBe(true);
    expect(carousel.neighbors()).toHaveLength(2);
    expect(carousel.neighbors().every(isAtRest)).toBe(true);
    // Once a neighbor's full image loads, its thumbnail underlay is gone, leaving exactly what the frame will show
    // (blue is the last image, so only the previous panel has one).
    await vi.waitFor(() =>
      expect(carousel.neighbors().map((neighbor) => neighbor.querySelectorAll('img').length)).toEqual([1, 0])
    );
  });

  it('commits a slow drag past the threshold, and returns a short one without navigating', async () => {
    const carousel = await renderCarousel();

    // 50px, lifted after resting: the release carries no velocity and stays under 40% of the stage.
    await swipe(carousel.media(), [100, 125, 150, 150], { stepMs: 150 });
    await vi.waitFor(() => expect(isAtRest(carousel.content())).toBe(true));

    expect(carousel.onNavigate).not.toHaveBeenCalled();

    // 140px the same way passes 40%.
    await swipe(carousel.media(), [60, 130, 200, 200], { stepMs: 150 });
    await vi.waitFor(() => expect(carousel.media().getAttribute('alt')).toBe('red'));

    expect(carousel.onNavigate).toHaveBeenCalledExactlyOnceWith(-1);
  });

  it('resists past the last image and settles back without navigating', async () => {
    const carousel = await renderCarousel({ start: 2 });
    const image = carousel.media();
    const base = performance.now();

    await interact(() => touch('pointerdown', image, 250, 140, base));
    await interact(() => touch('pointermove', document, 150, 140, base + 16));
    await interact(() => touch('pointermove', document, 50, 140, base + 32));

    // 200px of finger travel with nothing to reveal moves the image well under a third of the stage.
    expect(offsetOf(carousel.content())).toBeLessThan(0);
    expect(offsetOf(carousel.content())).toBeGreaterThan(-STAGE_WIDTH * 0.3);

    await interact(() => touch('pointerup', document, 50, 140, base + 48));
    await vi.waitFor(() => expect(isAtRest(carousel.content())).toBe(true));

    expect(carousel.onNavigate).not.toHaveBeenCalled();
  });

  it('slides the image back when the step does not happen', async () => {
    const carousel = await renderCarousel({ navigateResult: false });

    await swipe(carousel.media(), [200, 170, 140]);
    await vi.waitFor(() => expect(carousel.onNavigate).toHaveBeenCalledOnce());
    await vi.waitFor(() => expect(isAtRest(carousel.content())).toBe(true));

    expect(carousel.media().getAttribute('alt')).toBe('green');
  });

  it('shows a loading neighbor for an unloaded page and returns if the step never lands', async () => {
    const carousel = await renderCarousel({ navigateResult: 'pending', next: { kind: 'more' } });

    await swipe(carousel.media(), [200, 170, 140]);

    const next = carousel.neighbors().find((neighbor) => neighbor.dataset.swipeNeighbor === 'next')!;

    expect(next.querySelector('.chakra-spinner')).not.toBeNull();
    await vi.waitFor(() => expect(offsetOf(carousel.content())).toBe(-STAGE_WIDTH));
    // The park is bounded even while the fetch is still in flight.
    await vi.waitFor(() => expect(isAtRest(carousel.content())).toBe(true), { timeout: 3000 });

    expect(carousel.onNavigate).toHaveBeenCalledOnce();
    expect(carousel.media().getAttribute('alt')).toBe('green');
  });

  it('sends a caught return home on a tap instead of committing it', async () => {
    const carousel = await renderCarousel();
    const base = performance.now();

    // Drag past the commit threshold, then flick back: the release cancels and the image starts home.
    await interact(() => touch('pointerdown', carousel.media(), 250, 140, base), 0);
    await interact(() => touch('pointermove', document, 40, 140, base + 150), 0);
    await interact(() => touch('pointermove', document, 80, 140, base + 166), 0);
    await interact(() => touch('pointerup', document, 80, 140, base + 182), 0);

    // Catch it on the way and lift without moving.
    await interact(() => touch('pointerdown', carousel.media(), 150, 140, base + 200), 0);
    await interact(() => touch('pointerup', document, 150, 140, base + 260));
    await vi.waitFor(() => expect(isAtRest(carousel.content())).toBe(true));

    expect(carousel.onNavigate).not.toHaveBeenCalled();
  });

  it('leaves a held touch to drag-and-drop, including one that catches a returning image', async () => {
    const carousel = await renderCarousel();

    // Moving before the hold elapses is a swipe, never a drag.
    await swipe(carousel.media(), [200, 170, 140]);
    await vi.waitFor(() => expect(carousel.media().getAttribute('alt')).toBe('blue'));

    expect(carousel.onDrop).not.toHaveBeenCalled();

    // A short drag starts the image home; a touch that catches it and holds arms the drag, which then wins.
    const image = carousel.media();
    const base = performance.now();

    await swipe(image, [150, 170, 175, 175], { stepMs: 150 });
    await interact(() => touch('pointerdown', image, 150, 140, base + 1000), HOLD_DELAY_MS + 100);
    await interact(() => touch('pointermove', document, 250, 120, base + 2000));
    await interact(() => touch('pointermove', document, 480, 100, base + 2016));
    // The armed drag sent the caught image home; the drag's own moves never move the track.
    await vi.waitFor(() => expect(isAtRest(carousel.content())).toBe(true));
    await interact(() => touch('pointermove', document, 482, 100, base + 2032));

    expect(isAtRest(carousel.content())).toBe(true);

    await interact(() => touch('pointerup', document, 482, 100, base + 2048));

    expect(carousel.onDrop).toHaveBeenCalledExactlyOnceWith('test-drop');
    expect(carousel.onNavigate).toHaveBeenCalledOnce();
  });

  it('pans a zoomed image instead of swiping', async () => {
    const carousel = await renderCarousel();
    const stage = carousel.content().parentElement!;

    await interact(() =>
      stage.dispatchEvent(
        new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: 190, clientY: 160, deltaY: -600 })
      )
    );
    await vi.waitFor(() => expect(carousel.content().style.transform).toContain('scale'));

    await swipe(carousel.media(), [200, 170, 140]);
    await interact(() => {}, 400);

    expect(carousel.onNavigate).not.toHaveBeenCalled();
    expect(isAtRest(carousel.content())).toBe(true);
  });

  it('ignores mouse drags, which keep dragging the image', async () => {
    const carousel = await renderCarousel();
    const image = carousel.media();
    const mouse = { pointerType: 'mouse' };

    await interact(() => touch('pointerdown', image, 150, 140, 0, mouse));
    image.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, button: 0, clientX: 150, clientY: 140 }));
    await interact(() => {
      touch('pointermove', document, 250, 120, 16, mouse);
      document.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 250, clientY: 120 }));
    });
    await interact(() => {
      touch('pointermove', document, 480, 100, 32, mouse);
      document.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 480, clientY: 100 }));
    });

    expect(isAtRest(carousel.content())).toBe(true);

    await interact(() => {
      touch('pointerup', document, 480, 100, 48, mouse);
      document.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, clientX: 480, clientY: 100 }));
    });

    expect(carousel.onNavigate).not.toHaveBeenCalled();
    expect(carousel.onDrop).toHaveBeenCalledExactlyOnceWith('test-drop');
  });

  it('hands a second finger to the pinch with the image back at rest', async () => {
    const carousel = await renderCarousel();
    const image = carousel.media();
    const base = performance.now();

    await interact(() => touch('pointerdown', image, 200, 140, base));
    await interact(() => touch('pointermove', document, 160, 140, base + 16));
    await interact(() => touch('pointerdown', image, 240, 140, base + 32, { isPrimary: false, pointerId: 2 }));

    expect(isAtRest(carousel.content())).toBe(true);

    await interact(() => {
      touch('pointerup', document, 100, 140, base + 48);
      touch('pointerup', document, 240, 140, base + 48, { isPrimary: false, pointerId: 2 });
    }, 400);

    expect(carousel.onNavigate).not.toHaveBeenCalled();
  });

  it('completes a video swipe that the browser or the player cancels midway', async () => {
    const carousel = await renderCarousel({ items: [makeItem('red'), makeItem('clip', 'video'), makeItem('blue')] });
    const video = carousel.media();
    const y = video.getBoundingClientRect().top + 20;
    const base = performance.now();

    // Nothing native is left to claim one-finger travel on the stage.
    expect(getComputedStyle(carousel.content().parentElement!).touchAction).toBe('none');

    await interact(() => touch('pointerdown', video, 200, y, base), 0);
    await interact(() => touch('pointermove', document, 170, y, base + 16), 0);
    await interact(() => touch('pointermove', document, 140, y, base + 32), 0);
    // The cancel arrives well after the last move; the flick it interrupted still counts.
    await interact(() => touch('pointercancel', document, 140, y, base + 400), 0);
    await vi.waitFor(() => expect(carousel.media().getAttribute('alt')).toBe('blue'));

    expect(carousel.onNavigate).toHaveBeenCalledExactlyOnceWith(1);
  });

  it('swipes a video from its picture but leaves the native control bar alone', async () => {
    const carousel = await renderCarousel({ items: [makeItem('red'), makeItem('clip', 'video'), makeItem('blue')] });
    const video = carousel.media();
    const rect = video.getBoundingClientRect();

    await swipe(video, [200, 170, 140], { y: rect.bottom - 10 });
    await interact(() => {}, 400);

    expect(carousel.onNavigate).not.toHaveBeenCalled();

    await swipe(video, [200, 170, 140], { y: rect.top + 20 });
    await vi.waitFor(() => expect(carousel.media().getAttribute('alt')).toBe('blue'));

    expect(carousel.onNavigate).toHaveBeenCalledExactlyOnceWith(1);
  });
});

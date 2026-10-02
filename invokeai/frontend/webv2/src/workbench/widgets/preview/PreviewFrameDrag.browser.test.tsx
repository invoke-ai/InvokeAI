/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { StreamingImageSource } from '@platform/ui/streaming-image/streamingImageSource';

import { ChakraProvider } from '@chakra-ui/react';
import {
  DndContext,
  PointerSensor,
  useDndMonitor,
  useDroppable,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core';
import { isGalleryImageDragData, isGalleryItemDragData } from '@features/gallery/utility';
import { system } from '@theme/system';
import { widgetCollisionDetection } from '@workbench/widgetDnd';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { PreviewFrame } from './PreviewFrame';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: { preview: { dragVideo: 'Drag video', dropToCompare: 'Drop to compare', resetZoom: 'Reset zoom' } },
      },
    },
  },
});

const source: StreamingImageSource = {
  alt: 'preview.png',
  height: 128,
  kind: 'fallback',
  src: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128"/>',
  width: 128,
};

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

const DropTarget = () => {
  const { setNodeRef } = useDroppable({ data: { kind: 'test-image-drop' }, id: 'test-image-drop' });

  return (
    <div
      ref={setNodeRef}
      data-testid="drop-target"
      style={{ height: 180, left: 320, position: 'fixed', top: 40, width: 180 }}
    />
  );
};

const DragMonitor = ({
  onDrop,
}: {
  onDrop: (result: { activeData: unknown; overId: string | number | null }) => void;
}) => {
  useDndMonitor({
    onDragEnd: (event: DragEndEvent) =>
      onDrop({ activeData: event.active.data.current, overId: event.over?.id ?? null }),
  });

  return null;
};

const renderHarness = async (media: 'image' | 'video' = 'image') => {
  const onDrop = vi.fn();
  const Harness = () => {
    const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }));

    return (
      <DndContext collisionDetection={widgetCollisionDetection} sensors={sensors}>
        <DragMonitor onDrop={onDrop} />
        <div style={{ display: 'flex', height: 220, left: 40, position: 'fixed', top: 40, width: 220 }}>
          {media === 'image' ? (
            <PreviewFrame
              dragItem={{ kind: 'image', name: 'preview.png' }}
              frameHeight={128}
              frameWidth={128}
              isLive={false}
              shouldAntialiasLiveImage
              source={{ itemKey: 'image:preview.png', kind: 'image', source }}
              variant="framed"
            />
          ) : (
            <PreviewFrame
              dragItem={{ kind: 'video', name: 'preview.mp4' }}
              frameHeight={128}
              frameWidth={128}
              isLive={false}
              shouldAntialiasLiveImage
              source={{
                itemKey: 'video:preview.mp4',
                kind: 'video',
                label: 'Video preview.mp4',
                poster: source.src,
                src: source.src,
              }}
              variant="framed"
            />
          )}
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

  return onDrop;
};

const pointer = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, clientX, clientY, isPrimary: true, pointerId: 1 })
  );
};

/** A non-primary touch — a second finger joining the one already down. */
const secondFinger = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, {
      bubbles: true,
      button: type === 'pointermove' ? -1 : 0,
      clientX,
      clientY,
      isPrimary: false,
      pointerId: 2,
      pointerType: 'touch',
    })
  );
};

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('PreviewFrame image drag', () => {
  it('emits the shared qualified single-image item payload accepted by image drop surfaces', async () => {
    const onDrop = await renderHarness();
    const image = host!.querySelector<HTMLImageElement>('img[alt="preview.png"]')!;

    await interact(() => pointer('pointerdown', image, 140, 140));
    await interact(() => pointer('pointermove', image.ownerDocument, 170, 140));
    await interact(() => pointer('pointermove', image.ownerDocument, 400, 120));
    await interact(() => pointer('pointerup', image.ownerDocument, 400, 120));

    expect(onDrop).toHaveBeenCalledOnce();
    const result = onDrop.mock.calls[0]?.[0];

    expect(result?.overId).toBe('test-image-drop');
    expect(isGalleryImageDragData(result?.activeData)).toBe(true);
    expect(result?.activeData).toEqual({
      items: [{ kind: 'image', name: 'preview.png' }],
      kind: 'gallery-item',
    });
  });

  it('gives a second finger the pinch and abandons the drag it interrupted', async () => {
    // A second touch must cancel image dragging before taking over pinch, avoiding an unintended drop on release.
    const onDrop = await renderHarness();
    const image = host!.querySelector<HTMLImageElement>('img[alt="preview.png"]')!;
    const frame = image.parentElement!;

    await interact(() => pointer('pointerdown', image, 140, 140));
    // Past the sensor's 6px activation distance: the drag is running.
    await interact(() => pointer('pointermove', image, 170, 140));
    await interact(() => secondFinger('pointerdown', image, 220, 140));
    // The fingers spread from 50px apart to 150px, so the pinch trebles the image.
    await interact(() => {
      pointer('pointermove', image, 150, 140);
      secondFinger('pointermove', image, 300, 140);
    });

    expect(frame.style.transform).toContain('scale(3)');

    await interact(() => {
      pointer('pointerup', image, 150, 140);
      secondFinger('pointerup', image, 300, 140);
    });

    expect(onDrop).not.toHaveBeenCalled();
  });

  it('does not offer its own compare target to the image it is showing', async () => {
    // Reject self-comparison drops so they cannot silently pause follow and arm a future comparison.
    const onDrop = await renderHarness();
    const image = host!.querySelector<HTMLImageElement>('img[alt="preview.png"]')!;

    await interact(() => pointer('pointerdown', image, 140, 140));
    await interact(() => pointer('pointermove', image.ownerDocument, 170, 140));

    expect(document.body.textContent).not.toContain('Drop to compare');

    await interact(() => pointer('pointermove', image.ownerDocument, 150, 150));
    await interact(() => pointer('pointerup', image.ownerDocument, 150, 150));

    expect(onDrop.mock.calls[0]?.[0]?.overId).toBeNull();
  });
});

describe('PreviewFrame video drag', () => {
  it('drags the shared gallery-item payload from the corner grip handle', async () => {
    const onDrop = await renderHarness('video');
    const handle = host!.querySelector<HTMLElement>('[title="Drag video"]')!;

    expect(handle).not.toBeNull();
    // Unfocusable drag surfaces prevent invisible keyboard drags and accidental Tab drops.
    expect(handle.tabIndex).toBeLessThan(0);
    expect(handle.closest('button')).toBeNull();
    const rect = handle.getBoundingClientRect();
    const startX = rect.left + rect.width / 2;
    const startY = rect.top + rect.height / 2;

    await interact(() => pointer('pointerdown', handle, startX, startY));
    await interact(() => pointer('pointermove', handle.ownerDocument, startX + 30, startY));
    await interact(() => pointer('pointermove', handle.ownerDocument, 410, 130));
    await interact(() => pointer('pointerup', handle.ownerDocument, 410, 130));

    expect(onDrop).toHaveBeenCalledOnce();
    const result = onDrop.mock.calls[0]?.[0];

    expect(result?.overId).toBe('test-image-drop');
    expect(isGalleryItemDragData(result?.activeData)).toBe(true);
    expect(result?.activeData).toEqual({
      items: [{ kind: 'video', name: 'preview.mp4' }],
      kind: 'gallery-item',
    });
  });
});

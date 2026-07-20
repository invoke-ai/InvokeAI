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
import { isGalleryImageDragData } from '@features/gallery/utility';
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
    en: { translation: { widgets: { preview: { dropToCompare: 'Drop to compare', resetZoom: 'Reset zoom' } } } },
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

const renderHarness = async () => {
  const onDrop = vi.fn();
  const Harness = () => {
    const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }));

    return (
      <DndContext collisionDetection={widgetCollisionDetection} sensors={sensors}>
        <DragMonitor onDrop={onDrop} />
        <div style={{ display: 'flex', height: 220, left: 40, position: 'fixed', top: 40, width: 220 }}>
          <PreviewFrame
            dragImage={{ boardId: 'board-1', imageName: 'preview.png' }}
            frameHeight={128}
            frameWidth={128}
            isLive={false}
            liveBadgeLabel="Generating"
            shouldAntialiasLiveImage
            source={source}
            variant="framed"
          />
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

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('PreviewFrame image drag', () => {
  it('emits the shared single-gallery-image payload accepted by image drop surfaces', async () => {
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
      images: [{ boardId: 'board-1', imageName: 'preview.png' }],
      kind: 'gallery-image',
    });
  });
});

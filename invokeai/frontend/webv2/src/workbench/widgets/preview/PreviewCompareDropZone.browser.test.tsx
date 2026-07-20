import { ChakraProvider } from '@chakra-ui/react';
/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import {
  DndContext,
  PointerSensor,
  useDndMonitor,
  useDraggable,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core';
import { getGalleryImageDragData, getGalleryImageDragId } from '@features/gallery/utility';
import { system } from '@theme/system';
import { widgetCollisionDetection } from '@workbench/widgetDnd';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { resolvePreviewCompareDrop } from './previewCompareDnd';
import { PreviewCompareDropZone } from './PreviewCompareDropZone';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: { translation: { widgets: { preview: { dropToCompare: 'Drop to compare' } } } },
  },
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

const DraggableThumb = () => {
  const { listeners, setNodeRef } = useDraggable({
    data: getGalleryImageDragData([{ boardId: 'none', imageName: 'dragged.png' }]),
    id: getGalleryImageDragId('dragged.png'),
  });

  return (
    <div
      ref={setNodeRef}
      {...listeners}
      data-testid="thumb"
      style={{ background: '#888', height: 40, left: 10, position: 'fixed', top: 10, touchAction: 'none', width: 40 }}
    />
  );
};

const DropMonitor = ({ onDrop }: { onDrop: (resolution: { imageName: string } | null) => void }) => {
  useDndMonitor({
    onDragEnd: (event: DragEndEvent) =>
      onDrop(resolvePreviewCompareDrop(event.active.data.current, event.over?.data.current ?? null)),
  });

  return null;
};

const renderHarness = async () => {
  const onDrop = vi.fn();
  const Harness = () => {
    const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }));

    return (
      <DndContext collisionDetection={widgetCollisionDetection} sensors={sensors}>
        <DropMonitor onDrop={onDrop} />
        <DraggableThumb />
        <div data-testid="frame" style={{ height: 300, left: 200, position: 'fixed', top: 100, width: 300 }}>
          <PreviewCompareDropZone />
        </div>
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

  return { onDrop };
};

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const pointer = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, clientX, clientY, isPrimary: true, pointerId: 1 })
  );
};

describe('PreviewCompareDropZone', () => {
  it('shows the drop ring during a gallery-image drag and resolves the drop', async () => {
    const { onDrop } = await renderHarness();
    const thumb = document.querySelector<HTMLElement>('[data-testid="thumb"]')!;

    expect(document.body.textContent).not.toContain('Drop to compare');

    // Start the drag (activation distance is 6px) and move over the frame.
    await interact(() => pointer('pointerdown', thumb, 30, 30));
    await interact(() => pointer('pointermove', thumb.ownerDocument, 60, 60));

    expect(document.body.textContent).toContain('Drop to compare');

    await interact(() => pointer('pointermove', thumb.ownerDocument, 350, 250));
    await interact(() => pointer('pointerup', thumb.ownerDocument, 350, 250));

    expect(onDrop).toHaveBeenCalledWith({ imageName: 'dragged.png' });
    expect(document.body.textContent).not.toContain('Drop to compare');
  });

  it('resolves null when the drop lands outside the zone', async () => {
    const { onDrop } = await renderHarness();
    const thumb = document.querySelector<HTMLElement>('[data-testid="thumb"]')!;

    await interact(() => pointer('pointerdown', thumb, 30, 30));
    await interact(() => pointer('pointermove', thumb.ownerDocument, 60, 60));
    await interact(() => pointer('pointermove', thumb.ownerDocument, 120, 400));
    await interact(() => pointer('pointerup', thumb.ownerDocument, 120, 400));

    expect(onDrop).toHaveBeenCalledWith(null);
  });
});

import { DndContext, PointerSensor, useDraggable, useSensor, useSensors } from '@dnd-kit/core';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { GalleryItemDragData } from './galleryDnd';

import { getGalleryItemDragData, useGalleryImageDroppable } from './galleryDnd';

const TARGET_IDS = [
  'upscale-input-image',
  'generate-reference-images',
  'regional-ref-image:layer-1:reference-1',
] as const;
const imageData = getGalleryItemDragData([{ kind: 'image', name: 'still.png' }]);
const videoData = getGalleryItemDragData([{ kind: 'video', name: 'clip.mp4' }]);
const mixedData = getGalleryItemDragData([
  { kind: 'image', name: 'still.png' },
  { kind: 'video', name: 'clip.mp4' },
]);

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const DragSource = ({ data }: { data: GalleryItemDragData }) => {
  const { listeners, setNodeRef } = useDraggable({ data, id: 'test-gallery-source' });

  return (
    <button
      ref={setNodeRef}
      {...listeners}
      data-testid="drag-source"
      style={{ height: 80, left: 20, position: 'fixed', top: 20, touchAction: 'none', width: 80 }}
      type="button"
    />
  );
};

const ImageDropTarget = ({ id }: { id: string }) => {
  const { isOver, setNodeRef } = useGalleryImageDroppable({ data: { kind: id }, id });

  return (
    <output
      ref={setNodeRef}
      data-over={String(isOver)}
      data-testid="drop-target"
      style={{ height: 120, left: 300, position: 'fixed', top: 20, width: 120 }}
    />
  );
};

const Harness = ({ data, targetId }: { data: GalleryItemDragData; targetId: string }) => {
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }));

  return (
    <DndContext sensors={sensors}>
      <DragSource data={data} />
      <ImageDropTarget id={targetId} />
    </DndContext>
  );
};

const interact = (action: () => void, delay = 50): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, delay);
    });
  });

const pointer = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, clientX, clientY, isPrimary: true, pointerId: 1 })
  );
};

const dragOverTarget = async (data: GalleryItemDragData, targetId: string): Promise<string | null> => {
  await interact(() => root?.render(<Harness data={data} targetId={targetId} />));
  const source = host!.querySelector<HTMLElement>('[data-testid="drag-source"]')!;

  await interact(() => pointer('pointerdown', source, 50, 50));
  await interact(() => pointer('pointermove', source.ownerDocument, 90, 50));
  await interact(() => pointer('pointermove', source.ownerDocument, 340, 60));

  const isOver = host!.querySelector('[data-testid="drop-target"]')?.getAttribute('data-over') ?? null;
  await interact(() => pointer('pointerup', source.ownerDocument, 340, 60));

  return isOver;
};

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await interact(() => root?.unmount(), 0);
  host?.remove();
  host = null;
  root = null;
});

describe.each(TARGET_IDS)('%s image-only drop presentation', (targetId) => {
  it('visually accepts an all-image gallery drag', async () => {
    expect(await dragOverTarget(imageData, targetId)).toBe('true');
  });

  it('does not visually accept a video gallery drag', async () => {
    expect(await dragOverTarget(videoData, targetId)).toBe('false');
  });

  it('does not visually accept a mixed gallery drag', async () => {
    expect(await dragOverTarget(mixedData, targetId)).toBe('false');
  });
});

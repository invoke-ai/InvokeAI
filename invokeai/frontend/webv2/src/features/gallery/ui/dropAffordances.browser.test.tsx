import type { GalleryItemRef } from '@features/gallery/core/items';

import { ChakraProvider } from '@chakra-ui/react';
/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { DndContext, PointerSensor, useDraggable, useSensor, useSensors } from '@dnd-kit/core';
import { DropTargetOverlay } from '@platform/ui/DropTargetOverlay';
import { system } from '@theme/system';
import { widgetCollisionDetection } from '@workbench/widgetDnd';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  getGalleryItemDragData,
  getGalleryItemDragId,
  isGalleryImageDragData,
  isSingleGalleryImageDragData,
  isSingleGalleryVideoDragData,
  useGalleryItemDroppable,
} from './galleryDnd';
import { GalleryDragCursor } from './GalleryDragCursor';

/**
 * The drag-in-flight affordances: while a compatible gallery drag is active
 * ANYWHERE, a target advertises itself (overlay + label) and the body carries
 * the closed-hand cursor flag; incompatible payloads (wrong kind, multi-item
 * on a single-item target, non-gallery drags) advertise nothing.
 */

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

const THUMB_STYLE = {
  background: '#888',
  height: 40,
  position: 'fixed',
  touchAction: 'none',
  width: 40,
} as const;

const DraggableThumb = ({ data, id, left, testId }: { data: unknown; id: string; left: number; testId: string }) => {
  const { listeners, setNodeRef } = useDraggable({ data: data as never, id });

  return <div ref={setNodeRef} {...listeners} data-testid={testId} style={{ ...THUMB_STYLE, left, top: 10 }} />;
};

const Target = ({
  accepts,
  id,
  label,
  left,
  shields,
}: {
  accepts: (data: unknown) => boolean;
  id: string;
  label: string;
  left: number;
  shields?: (data: unknown) => boolean;
}) => {
  const { acceptsActiveDrag, isOver, setNodeRef } = useGalleryItemDroppable(accepts, { id }, shields ?? accepts);

  return (
    <div ref={setNodeRef} data-testid={id} style={{ height: 120, left, position: 'fixed', top: 200, width: 160 }}>
      <DropTargetOverlay isActive={acceptsActiveDrag} isOver={isOver} label={label} />
    </div>
  );
};

const IMAGE_REF: GalleryItemRef = { kind: 'image', name: 'frame.png' };
const VIDEO_REF: GalleryItemRef = { kind: 'video', name: 'clip.mp4' };

const renderHarness = async () => {
  const onDragEnd = vi.fn();
  const Harness = () => {
    const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }));

    return (
      <DndContext collisionDetection={widgetCollisionDetection} sensors={sensors} onDragEnd={onDragEnd}>
        <GalleryDragCursor />
        <DraggableThumb
          data={getGalleryItemDragData([IMAGE_REF])}
          id={getGalleryItemDragId(IMAGE_REF, 'preview-frame')}
          left={10}
          testId="image-thumb"
        />
        <DraggableThumb
          data={getGalleryItemDragData([VIDEO_REF])}
          id={getGalleryItemDragId(VIDEO_REF, 'preview-frame')}
          left={60}
          testId="video-thumb"
        />
        <DraggableThumb
          data={getGalleryItemDragData([IMAGE_REF, { kind: 'image', name: 'second.png' }])}
          id="multi-image-drag"
          left={110}
          testId="multi-thumb"
        />
        <DraggableThumb data={{ kind: 'widget-instance' }} id="widget-drag" left={160} testId="widget-thumb" />
        <Target
          accepts={isSingleGalleryImageDragData}
          id="frame-target"
          label="Drop First Frame"
          left={200}
          shields={isGalleryImageDragData}
        />
        <Target accepts={isSingleGalleryVideoDragData} id="clip-target" label="Drop Initial Video" left={400} />
      </DndContext>
    );
  };

  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await interact(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Harness />
      </ChakraProvider>
    );
  });

  return { onDragEnd };
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

const dragFrom = async (testId: string): Promise<HTMLElement> => {
  const thumb = document.querySelector<HTMLElement>(`[data-testid="${testId}"]`)!;
  const rect = thumb.getBoundingClientRect();

  await interact(() => pointer('pointerdown', thumb, rect.left + 20, rect.top + 20));
  await interact(() => pointer('pointermove', thumb.ownerDocument, rect.left + 50, rect.top + 50));

  return thumb;
};

const release = async (thumb: HTMLElement, clientX: number, clientY: number): Promise<void> => {
  await interact(() => pointer('pointermove', thumb.ownerDocument, clientX, clientY));
  await interact(() => pointer('pointerup', thumb.ownerDocument, clientX, clientY));
};

describe('drag-in-flight drop affordances', () => {
  it('advertises only the kind-matched target during a drag, and flags the body cursor', async () => {
    await renderHarness();

    expect(document.body.textContent).not.toContain('Drop First Frame');
    expect(document.body.hasAttribute('data-gallery-drag')).toBe(false);

    // A single-image drag: the frame target advertises, the clip target does not.
    const imageThumb = await dragFrom('image-thumb');

    expect(document.body.textContent).toContain('Drop First Frame');
    expect(document.body.textContent).not.toContain('Drop Initial Video');
    expect(document.body.hasAttribute('data-gallery-drag')).toBe(true);

    await release(imageThumb, 120, 400);

    expect(document.body.textContent).not.toContain('Drop First Frame');
    expect(document.body.hasAttribute('data-gallery-drag')).toBe(false);

    // A single-video drag: the reverse.
    const videoThumb = await dragFrom('video-thumb');

    expect(document.body.textContent).toContain('Drop Initial Video');
    expect(document.body.textContent).not.toContain('Drop First Frame');
    expect(document.body.hasAttribute('data-gallery-drag')).toBe(true);

    await release(videoThumb, 120, 400);

    expect(document.body.hasAttribute('data-gallery-drag')).toBe(false);
  });

  it('offers a single-item target nothing for a multi-item drag, but still shields the release', async () => {
    const { onDragEnd } = await renderHarness();

    const multiThumb = await dragFrom('multi-thumb');

    expect(document.body.textContent).not.toContain('Drop First Frame');
    expect(document.body.textContent).not.toContain('Drop Initial Video');
    // Still a gallery drag: the cursor flag applies even with no eligible target.
    expect(document.body.hasAttribute('data-gallery-drag')).toBe(true);

    // Released over the frame target: the droppable stays armed (a dead drop,
    // matching pre-affordance behavior) rather than disappearing from the
    // collision candidates and handing the release to whatever is underneath.
    await release(multiThumb, 280, 260);

    expect(onDragEnd).toHaveBeenCalled();
    expect(onDragEnd.mock.calls.at(-1)?.[0]?.over?.id).toBe('frame-target');
  });

  it('ignores non-gallery drags entirely', async () => {
    await renderHarness();

    const widgetThumb = await dragFrom('widget-thumb');

    expect(document.body.textContent).not.toContain('Drop First Frame');
    expect(document.body.textContent).not.toContain('Drop Initial Video');
    expect(document.body.hasAttribute('data-gallery-drag')).toBe(false);

    await release(widgetThumb, 120, 400);
  });
});

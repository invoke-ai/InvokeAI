import { getClientRect, type ClientRect, type CollisionDetection, type DroppableContainer } from '@dnd-kit/core';
import { afterEach, describe, expect, it } from 'vitest';

import { measureDroppableVisibleRect, widgetCollisionDetection } from './widgetDnd';

let host: HTMLDivElement | null = null;

const mount = (html: string): HTMLDivElement => {
  host = document.createElement('div');
  host.style.cssText = 'left:0;position:fixed;top:0;';
  host.innerHTML = html;
  document.body.append(host);

  return host;
};

afterEach(() => {
  host?.remove();
  host = null;
});

describe('measureDroppableVisibleRect', () => {
  it('returns the raw client rect when nothing clips the droppable', () => {
    const root = mount('<div id="target" style="height:40px;width:120px;"></div>');
    const target = root.querySelector<HTMLElement>('#target')!;

    expect(measureDroppableVisibleRect(target)).toMatchObject({ height: 40, width: 120 });
  });

  it('collapses a droppable scrolled out of its container to zero height', () => {
    // Mirrors a gallery board row below the fold: its client rect would
    // otherwise extend past the list over whatever is rendered below.
    const root = mount(`
      <div id="scroller" style="height:100px;overflow-y:auto;width:200px;">
        <div style="height:120px;"></div>
        <div id="target" style="height:40px;"></div>
      </div>
    `);
    const target = root.querySelector<HTMLElement>('#target')!;
    const scroller = root.querySelector<HTMLElement>('#scroller')!;
    const rect = measureDroppableVisibleRect(target);
    const clip = scroller.getBoundingClientRect();

    expect(rect.height).toBe(0);
    expect(rect.top).toBe(clip.bottom);
    expect(rect.bottom).toBe(clip.bottom);
    // Width axis is not clipped by an overflow-y container.
    expect(rect.width).toBe(scroller.clientWidth);
  });

  it('keeps only the visible slice of a partially scrolled droppable', () => {
    const root = mount(`
      <div id="scroller" style="height:100px;overflow-y:auto;width:200px;">
        <div style="height:80px;"></div>
        <div id="target" style="height:40px;"></div>
      </div>
    `);
    const target = root.querySelector<HTMLElement>('#target')!;
    const scroller = root.querySelector<HTMLElement>('#scroller')!;
    const rect = measureDroppableVisibleRect(target);
    const clip = scroller.getBoundingClientRect();

    expect(rect.height).toBe(20);
    expect(rect.bottom).toBe(clip.bottom);
  });

  it('clips against every overflow ancestor, not just the nearest', () => {
    // The stacked gallery wraps the board panel in an overflow:hidden box
    // that is itself shorter than the scroll area it contains.
    const root = mount(`
      <div style="height:60px;overflow:hidden;width:200px;">
        <div id="scroller" style="height:100px;overflow-y:auto;">
          <div style="height:70px;"></div>
          <div id="target" style="height:40px;"></div>
        </div>
      </div>
    `);
    const target = root.querySelector<HTMLElement>('#target')!;
    const rect = measureDroppableVisibleRect(target);

    expect(rect.height).toBe(0);
  });

  it('tracks the container scrolling the droppable into view', () => {
    const root = mount(`
      <div id="scroller" style="height:100px;overflow-y:auto;width:200px;">
        <div style="height:120px;"></div>
        <div id="target" style="height:40px;"></div>
      </div>
    `);
    const target = root.querySelector<HTMLElement>('#target')!;
    const scroller = root.querySelector<HTMLElement>('#scroller')!;

    scroller.scrollTop = 60;

    expect(measureDroppableVisibleRect(target).height).toBe(40);
  });
});

describe('widgetCollisionDetection pointer visibility', () => {
  // dnd-kit measures droppable rects once at drag start and only shifts them
  // by ancestor scroll deltas afterwards, so these tests hand the detection
  // the stale drag-start rect while the live DOM says otherwise.
  const createCollisionArgs = (options: {
    droppables: Array<{ id: string; node: HTMLElement; rect: ClientRect }>;
    pointerCoordinates: { x: number; y: number };
  }): Parameters<CollisionDetection>[0] => {
    const { x, y } = options.pointerCoordinates;
    const collisionRect = { bottom: y + 5, height: 10, left: x - 5, right: x + 5, top: y - 5, width: 10 };
    const droppableContainers: DroppableContainer[] = options.droppables.map(({ id, node, rect }) => ({
      data: { current: { kind: 'test-target' } },
      disabled: false,
      id,
      key: id,
      node: { current: node },
      rect: { current: rect },
    }));

    return {
      active: {
        data: { current: { kind: 'test-drag' } },
        id: 'active',
        rect: { current: { initial: collisionRect, translated: collisionRect } },
      },
      collisionRect,
      droppableContainers,
      droppableRects: new Map(options.droppables.map(({ id, rect }) => [id, rect])),
      pointerCoordinates: options.pointerCoordinates,
    };
  };

  const mountBoardListLikeScroller = (): { scroller: HTMLElement; target: HTMLElement } => {
    const root = mount(`
      <div id="scroller" style="height:100px;overflow-y:auto;width:200px;">
        <div style="height:120px;"></div>
        <div id="target" style="height:40px;"></div>
      </div>
    `);

    return {
      scroller: root.querySelector<HTMLElement>('#scroller')!,
      target: root.querySelector<HTMLElement>('#target')!,
    };
  };

  it('ignores a pointer over the hidden part of a droppable scrolled out of view', () => {
    const { target } = mountBoardListLikeScroller();
    // Drag-start rect: unclipped, extending below the 100px scroller.
    const staleRect = getClientRect(target);
    const collisions = widgetCollisionDetection(
      createCollisionArgs({
        droppables: [{ id: 'board-row', node: target, rect: staleRect }],
        pointerCoordinates: { x: 100, y: (staleRect.top + staleRect.bottom) / 2 },
      })
    );

    expect(collisions).toEqual([]);
  });

  it('hits a droppable revealed by scrolling mid-drag despite its stale drag-start rect', () => {
    const { scroller, target } = mountBoardListLikeScroller();
    const dragStartRect = getClientRect(target);

    scroller.scrollTop = 60;

    // What dnd-kit's Rect getters report after the scroll: edges shifted by
    // the delta, width/height still frozen from drag start.
    const adjustedRect = { ...dragStartRect, bottom: dragStartRect.bottom - 60, top: dragStartRect.top - 60 };
    const collisions = widgetCollisionDetection(
      createCollisionArgs({
        droppables: [{ id: 'board-row', node: target, rect: adjustedRect }],
        pointerCoordinates: { x: 100, y: (adjustedRect.top + adjustedRect.bottom) / 2 },
      })
    );

    expect(collisions.map(({ id }) => id)).toEqual(['board-row']);
  });

  it('keeps only the visibly hit droppable when a hidden row overlays another target', () => {
    const { target } = mountBoardListLikeScroller();
    const staleRect = getClientRect(target);
    const grid = document.createElement('div');

    grid.style.cssText = 'height:200px;left:0;position:fixed;top:100px;width:200px;';
    host!.append(grid);

    const pointer = { x: 100, y: (staleRect.top + staleRect.bottom) / 2 };
    const collisions = widgetCollisionDetection(
      createCollisionArgs({
        droppables: [
          { id: 'board-row', node: target, rect: staleRect },
          { id: 'image-grid', node: grid, rect: getClientRect(grid) },
        ],
        pointerCoordinates: pointer,
      })
    );

    expect(collisions.map(({ id }) => id)).toEqual(['image-grid']);
  });
});

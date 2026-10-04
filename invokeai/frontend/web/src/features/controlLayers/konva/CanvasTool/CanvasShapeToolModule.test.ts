import { atom } from 'nanostores';
import { describe, expect, it, vi } from 'vitest';

vi.mock('konva', () => {
  class Node {
    add = vi.fn();
    visible = vi.fn();
    setAttrs = vi.fn();
  }
  return { default: { Group: Node, Path: Node, Circle: Node } };
});

import type { CanvasBezierPathState } from 'features/controlLayers/store/types';
import { approximateBezierPath } from 'features/controlLayers/util/bezierPath';
import type { KonvaEventObject } from 'konva/lib/Node';

import { CanvasShapeToolModule } from './CanvasShapeToolModule';
import type { CanvasToolModule } from './CanvasToolModule';

const setup = (
  type: 'vector_layer' | 'raster_layer' = 'vector_layer',
  shapeType: 'polygon' | 'freehand' = 'polygon',
  scale = 1
) => {
  const entity = {
    entityIdentifier: { type, id: 'layer' },
    state: { type, position: { x: 0, y: 0 } },
    bufferRenderer: { clearBuffer: vi.fn(), setBuffer: vi.fn(), commitBuffer: vi.fn() },
  };
  const parent = {
    $cursorPos: atom({ relative: { x: 0, y: 0 } }),
    $isPrimaryPointerDown: atom(true),
    manager: {
      buildPath: () => [],
      buildLogger: () => ({ debug: vi.fn() }),
      stage: { unscale: (n: number) => n / scale },
      getAdapter: () => entity,
      tool: { tools: { path: { addPathToEditSession: () => false } } },
      stateApi: {
        $altKey: atom(false),
        $ctrlKey: atom(false),
        $metaKey: atom(false),
        $shiftKey: atom(false),
        createStoreSubscription: () => () => {},
        getSelectedEntityAdapter: () => entity,
        getSettings: () => ({ shapeType }),
        getCurrentColor: () => ({ r: 0, g: 0, b: 0, a: 1 }),
        addVectorPath: vi.fn(),
      },
    },
  };
  const module = new CanvasShapeToolModule(parent as unknown as CanvasToolModule);
  vi.spyOn(module, 'render').mockImplementation(() => {});
  const click = async (x: number, y: number) => {
    parent.$cursorPos.set({ relative: { x, y } });
    await module.onStagePointerDown({
      evt: { button: 0, detail: 0, shiftKey: false },
    } as KonvaEventObject<PointerEvent>);
  };
  const move = async (x: number, y: number, outsideCanvas = false) => {
    parent.$cursorPos.set({ relative: { x, y } });
    if (outsideCanvas) {
      await module.onWindowPointerMove();
    } else {
      await module.onStagePointerMove({ evt: { shiftKey: false } } as KonvaEventObject<PointerEvent>);
    }
  };
  return { module, parent, entity, click, move };
};

describe('vector freehand at high zoom', () => {
  it.each([10, 20])('preserves subpixel samples and endpoints at scale %s', async (scale) => {
    const { module, parent, entity, click, move } = setup('vector_layer', 'freehand', scale);
    entity.state.position = { x: 100, y: -50 };
    await click(100.125, -49.625);
    await move(100.375, -49.5);
    expect(module.repr().freehandPoints.at(-1)).toEqual({ x: 0.375, y: 0.5 });
    await move(100.625, -49.375, true);
    expect(module.repr().freehandPoints[0]).toEqual({ x: 0.125, y: 0.375 });
    expect(module.repr().freehandPoints.at(-1)).toEqual({ x: 0.625, y: 0.625 });
    await module.onWindowPointerUp();
    expect(parent.manager.stateApi.addVectorPath).toHaveBeenCalledOnce();
    const path = parent.manager.stateApi.addVectorPath.mock.calls[0]![0].path as CanvasBezierPathState;
    expect(path.points).toHaveLength(2);
    expect(path.points[0]!.anchor).toEqual({ x: 0.125, y: 0.375 });
    expect(path.points[1]!.anchor).toEqual({ x: 0.625, y: 0.625 });
    expect(module.hasActiveSession()).toBe(false);
  });

  it.each([1, 2, 10, 20])('fits a noisy arc to a compact curve at scale %s', async (scale) => {
    const { module, parent, click, move } = setup('vector_layer', 'freehand', scale);
    const center = { x: 20.125, y: 20.375 };
    await click(center.x + 10, center.y);
    for (let i = 1; i <= 400; i++) {
      const angle = (Math.PI * i) / 400;
      const radius = 10 + 0.1 * Math.sin(angle * 40);
      await move(center.x + radius * Math.cos(angle), center.y + radius * Math.sin(angle));
    }
    const sampleCount = module.repr().freehandPoints.length;
    await module.onWindowPointerUp();
    expect(parent.manager.stateApi.addVectorPath).toHaveBeenCalledOnce();
    const path = parent.manager.stateApi.addVectorPath.mock.calls[0]![0].path as CanvasBezierPathState;
    expect(path.isClosed).toBe(false);
    expect(path.points.length).toBeGreaterThanOrEqual(2);
    expect(path.points.length).toBeLessThanOrEqual(8);
    expect(path.points.length).toBeLessThan(sampleCount / 3);
    for (const point of approximateBezierPath(path.points, false, 100)) {
      const radialError = Math.abs(Math.hypot(point.x - center.x, point.y - center.y) - 10);
      expect(radialError).toBeLessThan(Math.max(0.6, 2 / scale));
    }
  });

  it('retains pixel snapping for raster freehand', async () => {
    const { module, entity, parent, click, move } = setup('raster_layer', 'freehand', 20);
    await click(10.25, 20.75);
    await move(11.25, 21.75);
    expect(module.repr().freehandPoints[0]).toEqual({ x: 10, y: 20 });
    expect(module.repr().freehandPoints.at(-1)).toEqual({ x: 11, y: 21 });
    await module.onWindowPointerUp();
    expect(parent.manager.stateApi.addVectorPath).not.toHaveBeenCalled();
    expect(entity.bufferRenderer.commitBuffer).toHaveBeenCalledOnce();
  });
});

describe('vector polyline completion', () => {
  it.each([0, 1])('finishes on double-click with one endpoint despite %s px jitter', async (jitter) => {
    const { module, parent, click } = setup();
    await click(0, 0);
    await click(100, 0);
    await click(200, 100);
    await click(200 + jitter, 100);
    await module.onStageDoubleClick();
    const add = parent.manager.stateApi.addVectorPath;
    expect(add).toHaveBeenCalledOnce();
    expect(add.mock.calls[0]![0].path.isClosed).toBe(false);
    expect(add.mock.calls[0]![0].path.points.map((point: { anchor: unknown }) => point.anchor)).toEqual([
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 200, y: 100 },
    ]);
    expect(module.hasActiveSession()).toBe(false);
  });

  it('does not complete a raster polygon on double-click', async () => {
    const { module, parent, click } = setup('raster_layer');
    await click(0, 0);
    await click(100, 0);
    await click(200, 100);
    await module.onStageDoubleClick();
    expect(parent.manager.stateApi.addVectorPath).not.toHaveBeenCalled();
    expect(module.hasActivePolygonSession()).toBe(true);
  });
});

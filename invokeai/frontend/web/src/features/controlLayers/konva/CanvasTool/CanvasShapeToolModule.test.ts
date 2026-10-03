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

import type { KonvaEventObject } from 'konva/lib/Node';

import { CanvasShapeToolModule } from './CanvasShapeToolModule';
import type { CanvasToolModule } from './CanvasToolModule';

const setup = (type: 'vector_layer' | 'raster_layer' = 'vector_layer') => {
  const entity = {
    entityIdentifier: { type, id: 'layer' },
    state: { type, position: { x: 0, y: 0 } },
    bufferRenderer: { clearBuffer: vi.fn(), setBuffer: vi.fn() },
  };
  const parent = {
    $cursorPos: atom({ relative: { x: 0, y: 0 } }),
    manager: {
      buildPath: () => [],
      buildLogger: () => ({ debug: vi.fn() }),
      stage: { unscale: (n: number) => n },
      getAdapter: () => entity,
      tool: { tools: { path: { addPathToEditSession: () => false } } },
      stateApi: {
        $altKey: atom(false),
        $ctrlKey: atom(false),
        $metaKey: atom(false),
        $shiftKey: atom(false),
        createStoreSubscription: () => () => {},
        getSelectedEntityAdapter: () => entity,
        getSettings: () => ({ shapeType: 'polygon' }),
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
  return { module, parent, click };
};

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

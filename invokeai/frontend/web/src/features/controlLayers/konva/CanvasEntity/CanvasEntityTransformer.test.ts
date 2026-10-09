import { Group } from 'konva/lib/Group';
import { atom } from 'nanostores';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('konva', async () => {
  const { Rect } = await import('konva/lib/shapes/Rect');
  class Node {
    on = vi.fn();
  }
  return { default: { Rect, Transformer: Node } };
});
vi.mock('./CanvasEntityVectorLayerRenderer', () => ({ CanvasEntityVectorLayerRenderer: class {} }));
vi.mock('features/nodes/util/graph/generation/Graph', () => ({ Graph: class {} }));
vi.mock('features/toast/toast', () => ({ toast: vi.fn() }));
vi.mock('services/api/endpoints/images', () => ({ uploadImage: vi.fn() }));
vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => prefix,
  getEmptyRect: () => ({ x: 0, y: 0, width: 0, height: 0 }),
  canvasToImageData: () => ({ data: new Uint8ClampedArray(400), width: 10, height: 10 }),
}));

vi.mock('./vectorPathTransformPreview', () => ({
  prepareVectorPathTransformPreview: () => ({
    activePathNode: { getClientRect: () => ({ x: 10, y: 20, width: 30, height: 40 }) },
    previewGroup: { hasChildren: () => false, destroy: vi.fn() },
  }),
}));

import { CanvasEntityTransformer } from './CanvasEntityTransformer';
import type { CanvasEntityAdapter } from './types';

describe('transform bbox synchronization', () => {
  afterEach(() => vi.useRealTimers());

  it('flushes the pending bbox and waits for the worker before starting a transform', async () => {
    vi.useFakeTimers();
    const manager = {
      buildPath: () => [],
      buildLogger: () => ({ debug: vi.fn(), trace: vi.fn() }),
      $isBusy: atom(false),
      stage: { $stageAttrs: atom({ scale: 1 }) },
      tool: { $tool: atom('move'), tools: { path: { $editSession: atom(null) } } },
      stateApi: {
        $shiftKey: atom(false),
        $transformingAdapter: atom<unknown>(null),
        createStoreSubscription: () => () => {},
      },
      worker: { requestBbox: vi.fn() },
    };
    const rect = { x: 0, y: 0, width: 1000, height: 600 };
    const parent = {
      manager,
      state: { type: 'raster_layer' },
      entityIdentifier: { type: 'raster_layer', id: 'layer' },
      konva: { layer: { add: vi.fn() } },
      getCanvas: () => ({}),
      $canvasCache: atom(null),
      renderer: {
        hasObjects: () => true,
        needsPixelBbox: () => true,
        konva: { objectGroup: { getClientRect: () => rect } },
      },
    };
    const transformer = new CanvasEntityTransformer(parent as unknown as CanvasEntityAdapter);
    vi.spyOn(transformer, 'syncInteractionState').mockImplementation(() => {});
    vi.spyOn(transformer, 'updateBbox').mockImplementation(() => {});
    transformer.$pixelRect.set({ x: 0, y: 0, width: 10, height: 10 });
    transformer.requestRectCalculation();
    const start = transformer.startTransform();
    await vi.advanceTimersByTimeAsync(0);
    expect(manager.worker.requestBbox).toHaveBeenCalledOnce();
    expect(transformer.$isTransforming.get()).toBe(false);
    expect(manager.stateApi.$transformingAdapter.get()).toBeNull();

    const callback = manager.worker.requestBbox.mock.calls[0]![1];
    callback({ minX: 0, minY: 0, maxX: 1000, maxY: 600 });
    await start;
    expect(transformer.$pixelRect.get()).toEqual(rect);
    expect(transformer.$isTransforming.get()).toBe(true);
    expect(manager.stateApi.$transformingAdapter.get()).toBe(parent);
    transformer.transformMutex.release();
    transformer.requestRectCalculation.cancel();
  });
});

describe('vector path transform nudging', () => {
  const setup = async (zoom = 1) => {
    const session = { entityIdentifier: { id: 'layer', type: 'vector_layer' }, activePathId: 'path' };
    const path = { $editSession: atom(session), applyActivePathTransform: vi.fn() };
    const manager = {
      buildPath: () => [],
      buildLogger: () => ({ debug: vi.fn(), trace: vi.fn() }),
      $isBusy: atom(false),
      stage: { $stageAttrs: atom({ scale: zoom }), unscale: (value: number) => value / zoom },
      tool: { $tool: atom('move'), tools: { path } },
      stateApi: {
        $shiftKey: atom(false),
        $transformingAdapter: atom<unknown>(null),
        createStoreSubscription: () => () => {},
        moveEntityBy: vi.fn(),
        transformVectorLayer: vi.fn(),
      },
    };
    const objectGroup = new Group();
    const parent = {
      manager,
      state: { type: 'vector_layer', position: { x: 100, y: 200 } },
      entityIdentifier: session.entityIdentifier,
      konva: { layer: { add: vi.fn() } },
      renderer: { konva: { objectGroup }, render: vi.fn(async () => {}) },
      bufferRenderer: { konva: { group: new Group() } },
    };
    const transformer = new CanvasEntityTransformer(parent as unknown as CanvasEntityAdapter);
    vi.spyOn(transformer, 'syncInteractionState').mockImplementation(() => {});
    vi.spyOn(transformer, '_setInteractionMode').mockImplementation(() => {});
    vi.spyOn(transformer, 'updateBbox').mockImplementation(() => transformer.updatePosition());
    // Keep asynchronous bbox calculation out of these geometry/transaction tests.
    vi.spyOn(transformer, 'calculateRect').mockImplementation(async () => {});
    await transformer.startTransform({ vectorPathId: 'path' });
    return { transformer, objectGroup, parent, manager, path, session };
  };

  it.each([1, 2, 20])('nudges in canvas pixels at zoom %s and applies only the path matrix', async (zoom) => {
    const { transformer, objectGroup, parent, manager, path, session } = await setup(zoom);
    transformer.nudgeBy({ x: 1, y: 0 });
    transformer.nudgeBy({ x: 1, y: 0 });
    transformer.nudgeBy({ x: 0, y: -1 });
    expect(transformer.konva.proxyRect.position()).toEqual({ x: 112, y: 219 });
    expect(objectGroup.getTransform().getMatrix()).toEqual([1, 0, 0, 1, 102, 199]);
    expect(manager.stateApi.moveEntityBy).not.toHaveBeenCalled();
    expect(path.applyActivePathTransform).not.toHaveBeenCalled();
    await transformer.applyTransform();
    expect(path.applyActivePathTransform).toHaveBeenCalledWith([1, 0, 0, 1, 102, 199]);
    expect(manager.stateApi.transformVectorLayer).not.toHaveBeenCalled();
    expect(parent.state.position).toEqual({ x: 100, y: 200 });
    expect(path.$editSession.get()).toBe(session);
    expect(transformer.getIsTransformingVectorPath()).toBe(false);
    transformer.requestRectCalculation.cancel();
  });

  it('keeps canvas-axis directions after rotation and mirroring, and cancels without committing', async () => {
    const { transformer, objectGroup, manager, path, session } = await setup();
    transformer.konva.proxyRect.setAttrs({ scaleX: -2, scaleY: 3, rotation: 90 });
    transformer.syncObjectGroupWithProxyRect();
    const before = [...objectGroup.getTransform().getMatrix()];
    transformer.nudgeBy({ x: -1, y: 1 });
    const after = objectGroup.getTransform().getMatrix();
    expect(after.slice(0, 4)).toEqual(before.slice(0, 4));
    expect(after[4]).toBeCloseTo(before[4]! - 1);
    expect(after[5]).toBeCloseTo(before[5]! + 1);
    transformer.stopTransform();
    expect(objectGroup.getTransform().getMatrix()).toEqual([1, 0, 0, 1, 100, 200]);
    expect(path.applyActivePathTransform).not.toHaveBeenCalled();
    expect(manager.stateApi.moveEntityBy).not.toHaveBeenCalled();
    expect(path.$editSession.get()).toBe(session);
    await Promise.resolve();
    transformer.requestRectCalculation.cancel();
  });

  it('does not change the preview while applying', async () => {
    const { transformer, objectGroup } = await setup();
    transformer.$isProcessing.set(true);
    transformer.nudgeBy({ x: 1, y: 1 });
    expect(objectGroup.getTransform().getMatrix()).toEqual([1, 0, 0, 1, 100, 200]);
    transformer.stopTransform();
    await Promise.resolve();
    transformer.requestRectCalculation.cancel();
  });
});

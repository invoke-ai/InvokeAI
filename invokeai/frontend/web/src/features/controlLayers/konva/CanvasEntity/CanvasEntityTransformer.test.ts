import { atom } from 'nanostores';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('konva', () => {
  class Node {
    on = vi.fn();
  }
  return { default: { Rect: Node, Transformer: Node } };
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

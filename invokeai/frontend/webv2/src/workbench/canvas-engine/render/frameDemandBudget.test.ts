import type { CanvasDocumentContractV2, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';

import { describe, expect, it, vi } from 'vitest';

import { calculateActiveFrameLayerIds } from './frameDemand';
import { createLayerCacheStore } from './layerCache';
import { createTestStubRasterBackend } from './raster.testStub';

const layer = (id: string, x: number): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: 100, imageName: `${id}.png`, width: 100 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x, y: 0 },
  type: 'raster',
});

describe('frame demand cache allocation', () => {
  it('allocates on reveal, evicts offscreen LRU, and re-allocates deterministically on return', () => {
    const stub = createTestStubRasterBackend();
    const createSurface = vi.fn(stub.createSurface);
    const caches = createLayerCacheStore({ ...stub, createSurface });
    const document: CanvasDocumentContractV2 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 1_000,
      layers: [layer('left', 0), layer('right', 500)],
      selectedLayerId: null,
      version: 2,
      width: 1_000,
    };
    const allocateDemand = (x: number): Set<string> => {
      const active = calculateActiveFrameLayerIds({
        document,
        viewport: { height: 200, width: 200, x, y: 0 },
      });
      for (const id of active) {
        caches.getOrCreate(id, 100, 100);
      }
      caches.evictHidden(active, 40_000);
      return active;
    };

    expect(allocateDemand(0)).toEqual(new Set(['left']));
    expect(createSurface).toHaveBeenCalledTimes(1);
    expect(caches.byteSize()).toBe(40_000);

    expect(allocateDemand(450)).toEqual(new Set(['right']));
    expect(createSurface).toHaveBeenCalledTimes(2);
    expect(caches.peek('left')).toBeUndefined();
    expect(caches.byteSize()).toBe(40_000);

    expect(allocateDemand(0)).toEqual(new Set(['left']));
    expect(createSurface).toHaveBeenCalledTimes(3);
    expect(caches.peek('right')).toBeUndefined();
    expect(caches.byteSize()).toBe(40_000);
  });
});

import type { CanvasRasterLayerContractV2 } from '@workbench/types';

import { createCanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { RasterController } from './rasterController';

describe('RasterController', () => {
  it('owns base and derived surfaces and invalidates all effects for a layer', () => {
    const backend = createTestStubRasterBackend();
    const onVersionChange = vi.fn();
    const controller = new RasterController({
      backend,
      diagnostics: createCanvasDiagnostics(true),
      onVersionChange,
    });
    const entry = controller.layers.getOrCreate('a', 10, 10);
    const layer = {
      adjustments: { brightness: 0.5, contrast: 0, saturation: 0 },
      id: 'a',
      type: 'raster',
    } as CanvasRasterLayerContractV2;
    expect(controller.getAdjustedSurface(layer, entry)).not.toBeNull();
    expect(controller.derived.byteSize()).toBe(400);

    controller.deleteDerivedSurfaces('a');
    expect(controller.derived.byteSize()).toBe(0);
    controller.dispose();
    controller.dispose();
    expect(controller.layers.byteSize()).toBe(0);
  });
});

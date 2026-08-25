import type { CompositeEntry } from '@workbench/canvas-operations/generationContracts';

import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

import { createCompositeDedupeCache, executeControlComposite } from './compositeForGeneration';

describe('executeControlComposite — real browser pixels', () => {
  it('uploads erased transparent pixels as opaque black', async () => {
    const backend = createDomRasterBackend();
    const layerSurface = backend.createSurface(3, 1);
    layerSurface.ctx.putImageData(
      new ImageData(
        new Uint8ClampedArray([
          255,
          255,
          255,
          255, // opaque white control edge
          120,
          80,
          40,
          128, // partially transparent antialiasing
          255,
          255,
          255,
          0, // erased pixel with non-black hidden RGB
        ]),
        3,
        1
      ),
      0,
      0
    );
    const entry: CompositeEntry = {
      bbox: { height: 1, width: 3, x: 0, y: 0 },
      key: 'control-layer|transparent-pixels',
      kind: 'control-layer',
      layerId: 'control',
      layers: [
        {
          blendMode: 'normal',
          contentOffset: { x: 0, y: 0 },
          contentSize: { height: 1, width: 3 },
          id: 'control',
          opacity: 1,
          sourceRef: 'paint:control.png',
          transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
        },
      ],
    };
    let uploadedPixels: number[] | null = null;

    await executeControlComposite(entry, {
      backend,
      dedupe: createCompositeDedupeCache(),
      getLayerSurface: () => Promise.resolve({ rect: { height: 1, width: 3, x: 0, y: 0 }, surface: layerSurface }),
      hashBlob: () => Promise.resolve('control-pixels'),
      uploadImage: async (blob) => {
        const bitmap = await backend.createImageBitmap(blob);
        const decoded = backend.createSurface(3, 1);
        decoded.ctx.drawImage(bitmap, 0, 0);
        uploadedPixels = [...decoded.ctx.getImageData(0, 0, 3, 1).data];
        bitmap.close();
        return { height: 1, imageName: 'control.png', width: 3 };
      },
    });

    expect(uploadedPixels).toEqual([255, 255, 255, 255, 60, 40, 20, 255, 0, 0, 0, 255]);
  });
});

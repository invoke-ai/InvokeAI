import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createSelectionState } from '@workbench/canvas-engine/selection/selectionState';
import { expect, it } from 'vitest';

it('replaceMask keeps exact partial coverage, isolated from later edits to the source surface', () => {
  const backend = createDomRasterBackend();
  const selection = createSelectionState({
    backend,
    createPath2D: (d) => new Path2D(d),
    getDocumentSize: () => ({ height: 100, width: 100 }),
    onChange: () => undefined,
  });
  const source = backend.createSurface(2, 2);
  const pixels = source.ctx.createImageData(2, 2);
  [0, 64, 255, 0].forEach((alpha, index) => {
    pixels.data.set([255, 255, 255, alpha], index * 4);
  });
  source.ctx.putImageData(pixels, 0, 0);

  selection.replaceMask({ rect: { height: 2, width: 2, x: 7, y: -3 }, surface: source });
  source.ctx.clearRect(0, 0, 2, 2);

  const mask = selection.mask()!;
  expect(mask.rect).toEqual({ height: 2, width: 2, x: 7, y: -3 });
  const alpha = [...mask.surface.ctx.getImageData(0, 0, 2, 2).data].filter((_, index) => index % 4 === 3);
  expect(alpha).toEqual([0, 64, 255, 0]);
});

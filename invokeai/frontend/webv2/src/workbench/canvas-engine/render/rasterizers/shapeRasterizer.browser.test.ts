import type { CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

import { drawShapeSource, rasterizeShapeSource } from './shapeRasterizer';

type ShapeSource = Extract<CanvasLayerSourceContract, { type: 'shape' }>;

const fixtures: readonly {
  name: string;
  source: ShapeSource;
  border: { x: number; y: number };
  interior: { x: number; y: number };
  outside: { x: number; y: number };
}[] = [
  {
    name: 'triangle',
    border: { x: 40, y: 8 },
    interior: { x: 40, y: 70 },
    outside: { x: 2, y: 10 },
    source: {
      fill: '#ff0000',
      height: 120,
      kind: 'triangle',
      stroke: '#ffffff',
      strokeWidth: 18,
      type: 'shape',
      width: 80,
    },
  },
  {
    name: 'concave polygon (a closed freehand shape)',
    border: { x: 2, y: 60 },
    interior: { x: 20, y: 75 },
    outside: { x: 60, y: 90 },
    source: {
      fill: '#ff0000',
      height: 120,
      kind: 'polygon',
      points: [
        { x: 0, y: 0 },
        { x: 120, y: 0 },
        { x: 120, y: 120 },
        { x: 80, y: 120 },
        { x: 70, y: 35 },
        { x: 50, y: 35 },
        { x: 40, y: 120 },
        { x: 0, y: 120 },
      ],
      stroke: '#ffffff',
      strokeWidth: 18,
      type: 'shape',
      width: 120,
    },
  },
  {
    name: 'rectangle',
    border: { x: 2, y: 50 },
    interior: { x: 60, y: 50 },
    outside: { x: -2, y: 50 },
    source: {
      fill: '#ff0000',
      height: 100,
      kind: 'rect',
      stroke: '#ffffff',
      strokeWidth: 18,
      type: 'shape',
      width: 120,
    },
  },
  {
    name: 'ellipse',
    border: { x: 60, y: 2 },
    interior: { x: 60, y: 50 },
    outside: { x: 2, y: 2 },
    source: {
      fill: '#ff0000',
      height: 100,
      kind: 'ellipse',
      stroke: '#ffffff',
      strokeWidth: 18,
      type: 'shape',
      width: 120,
    },
  },
  {
    name: 'star',
    border: { x: 80, y: 10 },
    interior: { x: 80, y: 80 },
    outside: { x: 10, y: 10 },
    source: {
      fill: '#ff0000',
      height: 160,
      kind: 'star',
      stroke: '#ffffff',
      strokeWidth: 18,
      type: 'shape',
      width: 160,
    },
  },
];

describe('shape borders in Chromium', () => {
  it.each([
    { name: 'triangle slope', source: fixtures[0]!.source, x: 38, y: 3 },
    { name: 'concave notch slope', source: fixtures[1]!.source, x: 74, y: 73 },
  ])('keeps the antialiased white outline free of red fill at the $name', ({ source, x, y }) => {
    const surface = createDomRasterBackend().createSurface(source.width + 32, source.height + 32);
    drawShapeSource(surface.ctx, source, 16, 16, source.width, source.height);
    const pixel = Array.from(surface.ctx.getImageData(x + 16, y + 16, 1, 1).data);
    expect(pixel[3]).toBeGreaterThan(0);
    expect(pixel[3]).toBeLessThan(255);
    expect(pixel.slice(0, 3), `Exterior RGBA: ${JSON.stringify(pixel)}`).toEqual([255, 255, 255]);
  });

  it.each(fixtures)('keeps the thick white border outside the red fill for $name', ({ source }) => {
    const backend = createDomRasterBackend();
    const surface = backend.createSurface(source.width + 32, source.height + 32);
    drawShapeSource(surface.ctx, source, 16, 16, source.width, source.height);
    const { data, height, width } = surface.ctx.getImageData(0, 0, surface.width, surface.height);
    const exposedFill: { x: number; y: number }[] = [];
    let redPixels = 0;
    let whitePixels = 0;
    for (let y = 2; y < height - 2; y += 1) {
      for (let x = 2; x < width - 2; x += 1) {
        const index = (y * width + x) * 4;
        if (data[index + 3]! < 240) {
          continue;
        }
        if (data[index]! > 240 && data[index + 1]! > 240 && data[index + 2]! > 240) {
          whitePixels += 1;
        }
        if (data[index]! < 240 || data[index + 1]! > 16 || data[index + 2]! > 16) {
          continue;
        }
        redPixels += 1;
        if (
          [
            [-2, 0],
            [2, 0],
            [0, -2],
            [0, 2],
          ].some(([dx, dy]) => data[((y + dy!) * width + x + dx!) * 4 + 3]! < 16)
        ) {
          exposedFill.push({ x: x - 16, y: y - 16 });
        }
      }
    }
    expect(redPixels).toBeGreaterThan(0);
    expect(whitePixels).toBeGreaterThan(0);
    expect(exposedFill.slice(0, 8)).toEqual([]);
  });

  it('clears a reused rasterization target after a stroked shape', async () => {
    const backend = createDomRasterBackend();
    const target = backend.createSurface(64, 64);
    const deps = {
      backend,
      documentSize: { width: 64, height: 64 },
      resolver: () => Promise.resolve(new Blob()),
      store: createLayerCacheStore(backend),
    };
    const source: ShapeSource = {
      fill: '#ff0000',
      height: 64,
      kind: 'rect',
      stroke: '#ffffff',
      strokeWidth: 10,
      type: 'shape',
      width: 64,
    };
    await rasterizeShapeSource(source, deps, target);
    expect(target.ctx.getImageData(0, 0, 1, 1).data[3]).toBe(255);
    const result = await rasterizeShapeSource(
      { ...source, kind: 'triangle', fill: '#00ff00', stroke: null },
      deps,
      target
    );
    expect(result.surface).toBe(target);
    expect(target.ctx.getImageData(0, 0, 1, 1).data[3]).toBe(0);
    expect(Array.from(target.ctx.getImageData(32, 40, 1, 1).data)).toEqual([0, 255, 0, 255]);
  });
});

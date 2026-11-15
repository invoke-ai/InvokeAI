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
    // These pixels straddle the known outer silhouette, far from the outline's inner color transition.
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
        // An 18px white border must separate solid red fill from transparency by more than antialiasing alone.
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

  it.each(fixtures)(
    'keeps opaque fill under translucent and transparent strokes for $name',
    ({ source, border, interior }) => {
      const backend = createDomRasterBackend();
      for (const [stroke, green] of [
        ['#ffffff80', 128],
        ['#ffffff00', 0],
      ] as const) {
        const surface = backend.createSurface(source.width + 32, source.height + 32);
        drawShapeSource(surface.ctx, { ...source, stroke }, 16, 16, source.width, source.height);
        const borderPixel = Array.from(surface.ctx.getImageData(border.x + 16, border.y + 16, 1, 1).data);
        expect(borderPixel).toEqual([255, green, green, 255]);
        expect(Array.from(surface.ctx.getImageData(interior.x + 16, interior.y + 16, 1, 1).data)).toEqual([
          255, 0, 0, 255,
        ]);
      }
    }
  );

  it.each(fixtures)(
    'contains stroke-only coverage within the original silhouette and extent for $name',
    ({ source, border, interior, outside }) => {
      const backend = createDomRasterBackend();
      const surface = backend.createSurface(source.width + 32, source.height + 32);
      drawShapeSource(surface.ctx, { ...source, fill: null }, 16, 16, source.width, source.height);
      expect(Array.from(surface.ctx.getImageData(border.x + 16, border.y + 16, 1, 1).data)).toEqual([
        255, 255, 255, 255,
      ]);
      expect(surface.ctx.getImageData(interior.x + 16, interior.y + 16, 1, 1).data[3]).toBe(0);
      expect(surface.ctx.getImageData(outside.x + 16, outside.y + 16, 1, 1).data[3]).toBe(0);
      const { data, width, height } = surface.ctx.getImageData(0, 0, surface.width, surface.height);
      let outsideAlpha = 0;
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          if (x < 16 || x >= source.width + 16 || y < 16 || y >= source.height + 16) {
            outsideAlpha += data[(y * width + x) * 4 + 3]!;
          }
        }
      }
      expect(outsideAlpha).toBe(0);
    }
  );

  it('uses the configured width inside the original rectangle edge', () => {
    const surface = createDomRasterBackend().createSurface(100, 100);
    drawShapeSource(
      surface.ctx,
      { fill: null, height: 60, kind: 'rect', stroke: '#ffffff', strokeWidth: 18, type: 'shape', width: 60 },
      20,
      20,
      60,
      60
    );
    expect(Array.from(surface.ctx.getImageData(37, 50, 1, 1).data)).toEqual([255, 255, 255, 255]);
    expect(surface.ctx.getImageData(38, 50, 1, 1).data[3]).toBe(0);
    expect(surface.ctx.getImageData(19, 50, 1, 1).data[3]).toBe(0);
  });

  it.each([
    ['#ffffff', [255, 255, 255, 255]],
    ['#ffffff80', [255, 170, 170, 192]],
  ] as const)('composites %s stroke independently over translucent fill', (stroke, borderPixel) => {
    const surface = createDomRasterBackend().createSurface(100, 100);
    drawShapeSource(
      surface.ctx,
      { fill: '#ff000080', height: 60, kind: 'rect', stroke, strokeWidth: 10, type: 'shape', width: 60 },
      20,
      20,
      60,
      60
    );
    expect(Array.from(surface.ctx.getImageData(22, 50, 1, 1).data)).toEqual(borderPixel);
    expect(Array.from(surface.ctx.getImageData(50, 50, 1, 1).data)).toEqual([255, 0, 0, 128]);
  });

  it('applies exterior coverage once with fractional placement and a nonuniform transform', () => {
    const surface = createDomRasterBackend().createSurface(160, 100);
    surface.ctx.setTransform(1.25, 0, 0, 0.75, 0.25, 0.25);
    drawShapeSource(
      surface.ctx,
      { fill: '#ff0000', height: 80, kind: 'rect', stroke: '#ffffff', strokeWidth: 10, type: 'shape', width: 80 },
      16,
      16,
      80,
      80
    );
    const pixel = Array.from(surface.ctx.getImageData(20, 12, 1, 1).data);
    expect(pixel[3]).toBeGreaterThan(0);
    expect(pixel[3]).toBeLessThan(255);
    expect(pixel.slice(0, 3)).toEqual([255, 255, 255]);
    expect(Array.from(surface.ctx.getImageData(70, 42, 1, 1).data)).toEqual([255, 0, 0, 255]);
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

  it('restores caller transform, alpha, composite mode and clip after drawing an outline', () => {
    const surface = createDomRasterBackend().createSurface(180, 180);
    const { ctx } = surface;
    ctx.beginPath();
    ctx.rect(0, 0, 150, 150);
    ctx.clip();
    ctx.setTransform(1, 0, 0, 1, 3, 2);
    ctx.globalAlpha = 0.5;
    ctx.globalCompositeOperation = 'xor';
    const { source } = fixtures[0]!;
    drawShapeSource(ctx, source, 16, 16, source.width, source.height);
    expect(ctx.globalAlpha).toBe(0.5);
    expect(ctx.globalCompositeOperation).toBe('xor');
    const { a, b, c, d, e, f } = ctx.getTransform();
    expect({ a, b, c, d, e, f }).toEqual({ a: 1, b: 0, c: 0, d: 1, e: 3, f: 2 });
    expect(ctx.getImageData(59, 88, 1, 1).data[3]).toBe(128);
    expect(Array.from(ctx.getImageData(59, 26, 1, 1).data)).toEqual([255, 170, 170, 192]);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = '#00ff00';
    ctx.fillRect(0, 0, 180, 180);
    expect(Array.from(ctx.getImageData(2, 2, 1, 1).data)).toEqual([0, 255, 0, 255]);
    expect(ctx.getImageData(160, 160, 1, 1).data[3]).toBe(0);
  });
});

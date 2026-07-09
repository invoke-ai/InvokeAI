import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { CanvasLayerSourceContract } from '@workbench/types';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it } from 'vitest';

import type { RasterizeDeps } from './types';

import { rasterizeShapeSource } from './shapeRasterizer';

type ShapeSource = Extract<CanvasLayerSourceContract, { type: 'shape' }>;

const makeDeps = (): RasterizeDeps => {
  const backend = createTestStubRasterBackend();
  return {
    backend,
    documentSize: { height: 200, width: 300 },
    resolver: () => Promise.resolve(new Blob()),
    store: createLayerCacheStore(backend),
  };
};

const ops = (surface: StubRasterSurface): string[] => surface.callLog.map((e) => e.op);

const rect = (over: Partial<ShapeSource> = {}): ShapeSource => ({
  fill: '#ff0000',
  height: 40,
  kind: 'rect',
  stroke: null,
  strokeWidth: 0,
  type: 'shape',
  width: 60,
  ...over,
});

describe('rasterizeShapeSource — extent + sizing', () => {
  it('sizes the surface to the source width/height (not the document)', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeShapeSource(rect(), deps)).surface as StubRasterSurface;
    expect(surface.width).toBe(60);
    expect(surface.height).toBe(40);
  });

  it('clears before drawing so re-rasterizing into a reused target is clean', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeShapeSource(rect(), deps)).surface as StubRasterSurface;
    expect(ops(surface)).toContain('clearRect');
  });
});

describe('rasterizeShapeSource — rect', () => {
  it('fills a rect covering the full extent when fill is set and no stroke', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeShapeSource(rect({ fill: '#00ff00', stroke: null }), deps))
      .surface as StubRasterSurface;
    const log = surface.callLog;
    // Fill style applied and a rect path filled.
    expect(log.some((e) => e.op === 'set' && e.args[0] === 'fillStyle' && e.args[1] === '#00ff00')).toBe(true);
    expect(ops(surface)).toContain('fill');
    // No stroke was drawn.
    expect(ops(surface)).not.toContain('stroke');
  });

  it('strokes with the stroke color + width, inset so it stays within the extent', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeShapeSource(rect({ fill: null, stroke: '#0000ff', strokeWidth: 8 }), deps))
      .surface as StubRasterSurface;
    const log = surface.callLog;
    expect(log.some((e) => e.op === 'set' && e.args[0] === 'strokeStyle' && e.args[1] === '#0000ff')).toBe(true);
    expect(log.some((e) => e.op === 'set' && e.args[0] === 'lineWidth' && e.args[1] === 8)).toBe(true);
    expect(ops(surface)).toContain('stroke');
    expect(ops(surface)).not.toContain('fill');
  });

  it('draws both fill and stroke when both are set', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeShapeSource(rect({ fill: '#111111', stroke: '#222222', strokeWidth: 4 }), deps))
      .surface as StubRasterSurface;
    expect(ops(surface)).toContain('fill');
    expect(ops(surface)).toContain('stroke');
  });

  it('draws nothing when both fill and stroke are null', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeShapeSource(rect({ fill: null, stroke: null }), deps)).surface as StubRasterSurface;
    expect(ops(surface)).not.toContain('fill');
    expect(ops(surface)).not.toContain('stroke');
  });
});

describe('rasterizeShapeSource — ellipse', () => {
  it('draws an ellipse path (not a rect) for the ellipse kind', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeShapeSource(rect({ fill: '#abcdef', kind: 'ellipse' }), deps))
      .surface as StubRasterSurface;
    expect(ops(surface)).toContain('ellipse');
    expect(ops(surface)).toContain('fill');
  });
});

describe('rasterizeShapeSource — target reuse', () => {
  it('resizes and reuses a provided target surface', async () => {
    const deps = makeDeps();
    const target = deps.backend.createSurface(10, 10);
    const { surface } = await rasterizeShapeSource(rect({ height: 40, width: 60 }), deps, target);
    expect(surface).toBe(target);
    expect(surface.width).toBe(60);
    expect(surface.height).toBe(40);
  });
});

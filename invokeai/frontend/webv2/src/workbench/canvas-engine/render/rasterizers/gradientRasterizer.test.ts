import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { CanvasLayerSourceContract } from '@workbench/types';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it } from 'vitest';

import type { RasterizeDeps } from './types';

import { rasterizeGradientSource } from './gradientRasterizer';

type GradientSource = Extract<CanvasLayerSourceContract, { type: 'gradient' }>;

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

const grad = (over: Partial<GradientSource> = {}): GradientSource => ({
  angle: 0,
  kind: 'linear',
  stops: [
    { color: '#000000', offset: 0 },
    { color: '#ffffff', offset: 1 },
  ],
  type: 'gradient',
  ...over,
});

describe('rasterizeGradientSource — extent', () => {
  it('defaults to the DOCUMENT extent for a legacy gradient (no explicit width/height)', async () => {
    const deps = makeDeps();
    const { rect, surface } = await rasterizeGradientSource(grad(), deps);
    expect(surface.width).toBe(300);
    expect(surface.height).toBe(200);
    expect(rect).toEqual({ height: 200, width: 300, x: 0, y: 0 });
    expect(ops(surface as StubRasterSurface)).toContain('fillRect');
  });

  it('uses the explicit width/height extent when present (content-sized)', async () => {
    const deps = makeDeps();
    const { rect, surface } = await rasterizeGradientSource(grad({ height: 90, width: 120 }), deps);
    expect(surface.width).toBe(120);
    expect(surface.height).toBe(90);
    expect(rect).toEqual({ height: 90, width: 120, x: 0, y: 0 });
  });
});

describe('rasterizeGradientSource — linear', () => {
  it('creates a linear gradient and applies every stop', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeGradientSource(grad({ angle: 90 }), deps)).surface as StubRasterSurface;
    expect(ops(surface)).toContain('createLinearGradient');
    const stops = surface.callLog.filter((e) => e.op === 'addColorStop');
    expect(stops).toHaveLength(2);
    expect(stops[0]?.args).toEqual([0, '#000000']);
    expect(stops[1]?.args).toEqual([1, '#ffffff']);
  });

  it('does not create a radial gradient for the linear kind', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeGradientSource(grad(), deps)).surface as StubRasterSurface;
    expect(ops(surface)).not.toContain('createRadialGradient');
  });
});

describe('rasterizeGradientSource — radial', () => {
  it('creates a radial gradient covering the document and applies stops', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeGradientSource(grad({ kind: 'radial' }), deps)).surface as StubRasterSurface;
    expect(ops(surface)).toContain('createRadialGradient');
    expect(ops(surface)).not.toContain('createLinearGradient');
    const stops = surface.callLog.filter((e) => e.op === 'addColorStop');
    expect(stops).toHaveLength(2);
  });

  it('supports more than two stops', async () => {
    const deps = makeDeps();
    const surface = (
      await rasterizeGradientSource(
        grad({
          stops: [
            { color: '#000000', offset: 0 },
            { color: '#ff0000', offset: 0.5 },
            { color: '#ffffff', offset: 1 },
          ],
        }),
        deps
      )
    ).surface as StubRasterSurface;
    const stops = surface.callLog.filter((e) => e.op === 'addColorStop');
    expect(stops).toHaveLength(3);
  });
});

describe('rasterizeGradientSource — target reuse', () => {
  it('resizes and reuses a provided target surface to the document size', async () => {
    const deps = makeDeps();
    const target = deps.backend.createSurface(10, 10);
    const { surface } = await rasterizeGradientSource(grad(), deps, target);
    expect(surface).toBe(target);
    expect(surface.width).toBe(300);
    expect(surface.height).toBe(200);
  });
});

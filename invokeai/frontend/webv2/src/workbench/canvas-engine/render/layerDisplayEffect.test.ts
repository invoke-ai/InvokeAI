import type { CanvasAdjustmentsContract, CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import { hasLayerDisplayEffect, renderLayerDisplayEffect } from './layerDisplayEffect';
import { createTestStubRasterBackend } from './raster.testStub';

const backend = createTestStubRasterBackend();

const controlLayer = (withTransparencyEffect: boolean): CanvasLayerContract => ({
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
  blendMode: 'normal',
  id: 'c',
  isEnabled: true,
  isLocked: false,
  name: 'c',
  opacity: 1,
  source: { image: { height: 10, imageName: 'c', width: 10 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect,
});

const rasterLayer = (adjustments?: CanvasAdjustmentsContract) =>
  ({
    adjustments,
    blendMode: 'normal',
    id: 'r',
    isEnabled: true,
    isLocked: false,
    name: 'r',
    opacity: 1,
    source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
  }) as CanvasLayerContract;

const maskLayer = (): CanvasLayerContract => ({
  blendMode: 'normal',
  id: 'm',
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: null, fill: { color: '#f00', style: 'solid' } },
  name: 'm',
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

describe('hasLayerDisplayEffect', () => {
  it('is true for a control layer with the transparency effect', () => {
    expect(hasLayerDisplayEffect(controlLayer(true))).toBe(true);
  });

  it('is false for a control layer without it', () => {
    expect(hasLayerDisplayEffect(controlLayer(false))).toBe(false);
  });

  it('is true for a raster layer with non-identity adjustments', () => {
    expect(
      hasLayerDisplayEffect(
        rasterLayer([{ brightness: 0.4, contrast: 0, id: 'b', isEnabled: true, type: 'brightness-contrast' }])
      )
    ).toBe(true);
  });

  it('is false for a raster layer with identity or absent adjustments', () => {
    expect(
      hasLayerDisplayEffect(
        rasterLayer([{ brightness: 0, contrast: 0, id: 'i', isEnabled: true, type: 'brightness-contrast' }])
      )
    ).toBe(false);
    expect(hasLayerDisplayEffect(rasterLayer(undefined))).toBe(false);
  });

  it('is false for mask layers — their fill is applied by the compositor, not per-pixel here', () => {
    expect(hasLayerDisplayEffect(maskLayer())).toBe(false);
  });
});

describe('renderLayerDisplayEffect', () => {
  it('returns null when the layer has no display effect, allocating nothing', () => {
    const source = backend.createSurface(4, 4);
    expect(renderLayerDisplayEffect(backend, controlLayer(false), source)).toBeNull();
  });

  it('returns a same-sized copy for a control layer with the transparency effect', () => {
    const source = backend.createSurface(6, 5);
    const out = renderLayerDisplayEffect(backend, controlLayer(true), source);

    expect(out).not.toBeNull();
    expect(out).not.toBe(source);
    expect({ height: out!.height, width: out!.width }).toEqual({ height: 5, width: 6 });
  });

  it('reads the source and writes the transformed pixels back', () => {
    const source = backend.createSurface(6, 5);
    const out = renderLayerDisplayEffect(backend, controlLayer(true), source)!;

    const log = (out as unknown as { callLog: { op: string; args: unknown[] }[] }).callLog;
    expect(log.some((entry) => entry.op === 'drawImage' && entry.args[0] === source.canvas)).toBe(true);
    expect(log.some((entry) => entry.op === 'getImageData')).toBe(true);
    expect(log.some((entry) => entry.op === 'putImageData')).toBe(true);
  });

  it('returns null for a zero-sized source', () => {
    expect(renderLayerDisplayEffect(backend, controlLayer(true), backend.createSurface(0, 0))).toBeNull();
  });

  it('returns a copy for a raster layer with adjustments', () => {
    const out = renderLayerDisplayEffect(
      backend,
      rasterLayer([{ brightness: 0.4, contrast: 0, id: 'b', isEnabled: true, type: 'brightness-contrast' }]),
      backend.createSurface(4, 4)
    );
    expect(out).not.toBeNull();
  });
});

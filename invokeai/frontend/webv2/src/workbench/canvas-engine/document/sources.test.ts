import type {
  CanvasDocumentContractV2,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import {
  getSourceBounds,
  getSourceContentRect,
  getSourcePixelSize,
  isMaskLayer,
  isRenderableLayer,
  maskAsPaintSource,
  renderableSourceOf,
} from './sources';

const doc = { height: 200, width: 300 } as CanvasDocumentContractV2;

const inpaintMask = (
  bitmap: { imageName: string; width: number; height: number } | null,
  offset?: { x: number; y: number }
): CanvasLayerContract =>
  ({
    blendMode: 'normal',
    id: 'm1',
    isEnabled: true,
    isLocked: false,
    mask: { bitmap, fill: { color: '#e07575', style: 'diagonal' }, ...(offset ? { offset } : {}) },
    name: 'M',
    opacity: 1,
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'inpaint_mask',
  }) as CanvasLayerContract;

describe('mask layers as paint-like sources', () => {
  it('isMaskLayer / renderableSourceOf treat a mask as an alpha paint source', () => {
    const layer = inpaintMask({ height: 30, imageName: 'mask', width: 40 }, { x: 5, y: 6 });
    expect(isMaskLayer(layer)).toBe(true);
    expect(maskAsPaintSource(layer)).toEqual({
      bitmap: { height: 30, imageName: 'mask', width: 40 },
      offset: { x: 5, y: 6 },
      type: 'paint',
    });
    expect(renderableSourceOf(layer)).toEqual(maskAsPaintSource(layer));
  });

  it('an enabled mask is renderable; content rect is the bitmap dims at its offset', () => {
    const layer = inpaintMask({ height: 30, imageName: 'mask', width: 40 }, { x: 5, y: 6 });
    expect(isRenderableLayer(layer)).toBe(true);
    expect(getSourceContentRect(layer, doc)).toEqual({ height: 30, width: 40, x: 5, y: 6 });
  });

  it('an empty (bitmap-less) mask is renderable but has an empty content rect', () => {
    const layer = inpaintMask(null);
    expect(isRenderableLayer(layer)).toBe(true);
    expect(getSourceContentRect(layer, doc)).toEqual({ height: 0, width: 0, x: 0, y: 0 });
  });

  it('a disabled mask is not renderable', () => {
    const layer = inpaintMask({ height: 30, imageName: 'mask', width: 40 });
    (layer as { isEnabled: boolean }).isEnabled = false;
    expect(isRenderableLayer(layer)).toBe(false);
  });
});

const rasterLayer = (source: CanvasLayerSourceContract, transformOver = {}): CanvasLayerContract =>
  ({
    blendMode: 'normal',
    id: 'l1',
    isEnabled: true,
    isLocked: false,
    name: 'L',
    opacity: 1,
    source,
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0, ...transformOver },
    type: 'raster',
  }) as CanvasLayerContract;

const shape: CanvasLayerSourceContract = {
  fill: '#000',
  height: 40,
  kind: 'rect',
  stroke: null,
  strokeWidth: 0,
  type: 'shape',
  width: 60,
};

const gradient: CanvasLayerSourceContract = {
  angle: 0,
  kind: 'linear',
  stops: [{ color: '#000', offset: 0 }],
  type: 'gradient',
};

describe('isRenderableLayer — parametric sources', () => {
  it('renders rect/ellipse shapes and gradients', () => {
    expect(isRenderableLayer(rasterLayer(shape))).toBe(true);
    expect(isRenderableLayer(rasterLayer({ ...shape, kind: 'ellipse' }))).toBe(true);
    expect(isRenderableLayer(rasterLayer(gradient))).toBe(true);
  });

  it('does not render a deferred polygon shape', () => {
    expect(isRenderableLayer(rasterLayer({ ...shape, kind: 'polygon' }))).toBe(false);
  });

  it('does not render a disabled layer', () => {
    expect(isRenderableLayer({ ...rasterLayer(shape), isEnabled: false })).toBe(false);
  });
});

describe('getSourcePixelSize — parametric sources', () => {
  it('sizes a shape to its own extent', () => {
    expect(getSourcePixelSize(rasterLayer(shape), doc)).toEqual({ height: 40, width: 60 });
  });

  it('sizes a gradient to the document', () => {
    expect(getSourcePixelSize(rasterLayer(gradient), doc)).toEqual({ height: 200, width: 300 });
  });
});

describe('getSourceBounds — parametric sources', () => {
  it('scales+offsets a shape by its transform', () => {
    const bounds = getSourceBounds(rasterLayer(shape, { scaleX: 2, scaleY: 3, x: 10, y: 20 }), doc);
    expect(bounds).toEqual({ height: 120, width: 120, x: 10, y: 20 });
  });

  it('returns the whole document for a gradient', () => {
    expect(getSourceBounds(rasterLayer(gradient), doc)).toEqual({ height: 200, width: 300, x: 0, y: 0 });
  });
});

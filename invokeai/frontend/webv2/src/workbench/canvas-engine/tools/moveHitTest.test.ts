import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import { estimateTextExtent } from '@workbench/canvas-engine/render/rasterizers/textRasterizer';
import { describe, expect, it } from 'vitest';

import { hitTestLayer, hittableLayerSize, layerOutlineCorners } from './moveHitTest';

const shapeLayer = (id: string, width: number, height: number): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { fill: '#ffffff', height, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const gradientLayer = (id: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { angle: 0, kind: 'linear', stops: [], type: 'gradient' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const textLayer = (id: string, content: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: {
    align: 'left',
    color: '#000000',
    content,
    fontFamily: 'sans',
    fontSize: 20,
    fontWeight: 400,
    lineHeight: 1.2,
    type: 'text',
  },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const imageLayer = (
  id: string,
  opts: {
    x?: number;
    y?: number;
    scaleX?: number;
    scaleY?: number;
    rotation?: number;
    width?: number;
    height?: number;
    isEnabled?: boolean;
    isLocked?: boolean;
  } = {}
): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: opts.isEnabled ?? true,
  isLocked: opts.isLocked ?? false,
  name: id,
  opacity: 1,
  source: { image: { height: opts.height ?? 20, imageName: id, width: opts.width ?? 20 }, type: 'image' },
  transform: {
    rotation: opts.rotation ?? 0,
    scaleX: opts.scaleX ?? 1,
    scaleY: opts.scaleY ?? 1,
    x: opts.x ?? 0,
    y: opts.y ?? 0,
  },
  type: 'raster',
});

const paintLayer = (
  id: string,
  bitmap?: { width: number; height: number; offset?: { x: number; y: number } }
): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: bitmap
    ? {
        bitmap: { height: bitmap.height, imageName: `${id}-bmp`, width: bitmap.width },
        offset: bitmap.offset,
        type: 'paint',
      }
    : { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const maskLayer = (
  id: string,
  bitmap?: { width: number; height: number; offset?: { x: number; y: number } }
): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: {
    bitmap: bitmap ? { height: bitmap.height, imageName: `${id}-mask`, width: bitmap.width } : null,
    fill: { color: '#ff0000', style: 'solid' },
    ...(bitmap?.offset ? { offset: bitmap.offset } : {}),
  },
  name: id,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

const doc = (layers: CanvasLayerContract[]): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers,
  selectedLayerId: null,
  version: 2,
  width: 100,
});

describe('hittableLayerSize', () => {
  it('returns native size for image layers', () => {
    expect(hittableLayerSize(imageLayer('a', { width: 30, height: 40 }), doc([]))).toEqual({ height: 40, width: 30 });
  });

  it('returns null for an EMPTY paint layer (content-sized: no bitmap ⇒ not hit-testable)', () => {
    expect(hittableLayerSize(paintLayer('a'), doc([]))).toBeNull();
  });

  it('returns the persisted bitmap size for a paint layer with content', () => {
    expect(hittableLayerSize(paintLayer('a', { height: 40, width: 60 }), doc([]))).toEqual({ height: 40, width: 60 });
  });

  it('returns null for an EMPTY (bitmap-less) mask, but the bitmap size for a painted mask (masks are movable)', () => {
    expect(hittableLayerSize(maskLayer('m'), doc([]))).toBeNull();
    expect(hittableLayerSize(maskLayer('m', { height: 24, width: 32 }), doc([]))).toEqual({ height: 24, width: 32 });
  });

  it('returns the shape extent for shape layers (param for parametric)', () => {
    expect(hittableLayerSize(shapeLayer('s', 30, 40), doc([]))).toEqual({ height: 40, width: 30 });
  });

  it('returns document size for gradient layers', () => {
    expect(hittableLayerSize(gradientLayer('g'), doc([]))).toEqual({ height: 100, width: 100 });
  });

  it('returns the estimated text extent for text layers', () => {
    const layer = textLayer('t', 'hi');
    expect(hittableLayerSize(layer, doc([]))).toEqual(
      estimateTextExtent(layer.type === 'raster' && layer.source.type === 'text' ? layer.source : (null as never))
    );
  });
});

describe('hitTestLayer: parametric layers are now hit-testable', () => {
  it('hits inside a shape layer and misses outside', () => {
    const layer = shapeLayer('s', 30, 40);
    const d = doc([layer]);
    expect(hitTestLayer(layer, d, { x: 15, y: 20 })).toBe(true);
    expect(hitTestLayer(layer, d, { x: 35, y: 20 })).toBe(false);
  });
});

describe('hitTestLayer', () => {
  it('hits inside the transformed bounds and misses outside', () => {
    const layer = imageLayer('a', { x: 10, y: 10, width: 20, height: 20 });
    const d = doc([layer]);
    expect(hitTestLayer(layer, d, { x: 15, y: 15 })).toBe(true);
    expect(hitTestLayer(layer, d, { x: 9, y: 15 })).toBe(false);
    expect(hitTestLayer(layer, d, { x: 31, y: 15 })).toBe(false);
  });

  it('accounts for scale', () => {
    // 20x20 image scaled x2 → covers [0,40]x[0,40] in document space.
    const layer = imageLayer('a', { scaleX: 2, scaleY: 2, width: 20, height: 20 });
    const d = doc([layer]);
    expect(hitTestLayer(layer, d, { x: 39, y: 39 })).toBe(true);
    expect(hitTestLayer(layer, d, { x: 41, y: 41 })).toBe(false);
  });

  it('accounts for rotation (90° about origin)', () => {
    // Rotate a 20x40 image 90° clockwise about its origin: local (0,0)->(0,0),
    // local (0,40) -> document (-40, 0). Document point (-10, 5) maps to a local
    // point inside [0,20]x[0,40].
    const layer = imageLayer('a', { rotation: Math.PI / 2, width: 20, height: 40 });
    const d = doc([layer]);
    expect(hitTestLayer(layer, d, { x: -10, y: 5 })).toBe(true);
    expect(hitTestLayer(layer, d, { x: 10, y: 5 })).toBe(false);
  });

  it('never hits a mask layer (no source bounds)', () => {
    const layer = maskLayer('m');
    expect(hitTestLayer(layer, doc([layer]), { x: 5, y: 5 })).toBe(false);
  });
});

describe('layerOutlineCorners', () => {
  it('returns the four document-space corners of the rendered rect', () => {
    const layer = imageLayer('a', { x: 5, y: 5, width: 10, height: 10 });
    const corners = layerOutlineCorners(layer, doc([layer]));
    expect(corners).toEqual([
      { x: 5, y: 5 },
      { x: 15, y: 5 },
      { x: 15, y: 15 },
      { x: 5, y: 15 },
    ]);
  });

  it('applies an x/y override without mutating the layer transform', () => {
    const layer = imageLayer('a', { x: 5, y: 5, width: 10, height: 10 });
    const corners = layerOutlineCorners(layer, doc([layer]), { x: 25, y: 30 });
    expect(corners?.[0]).toEqual({ x: 25, y: 30 });
    expect(corners?.[2]).toEqual({ x: 35, y: 40 });
    expect(layer.transform.x).toBe(5);
  });

  it('returns null for a mask layer', () => {
    const layer = maskLayer('m');
    expect(layerOutlineCorners(layer, doc([layer]))).toBeNull();
  });
});

describe('live cache rect (freshly-painted, not-yet-flushed content)', () => {
  it('hitTestLayer misses an empty paint layer with no live rect', () => {
    const layer = paintLayer('p'); // bitmap: null → empty persisted content
    expect(hitTestLayer(layer, doc([layer]), { x: 30, y: 30 })).toBe(false);
  });

  it('hitTestLayer hits content present only in the live cache rect', () => {
    const layer = paintLayer('p'); // persisted content still empty (unflushed stroke)
    const liveRect = { height: 40, width: 40, x: 20, y: 20 };
    // Inside the live rect: grabbable even though the contract bitmap is null.
    expect(hitTestLayer(layer, doc([layer]), { x: 30, y: 30 }, liveRect)).toBe(true);
    // Outside the live rect: still a miss.
    expect(hitTestLayer(layer, doc([layer]), { x: 5, y: 5 }, liveRect)).toBe(false);
  });

  it('unions the live rect with persisted content (both contribute to the hit area)', () => {
    const layer = paintLayer('p', { width: 10, height: 10 }); // persisted content [0,0,10,10]
    const liveRect = { height: 10, width: 10, x: 40, y: 40 }; // grown region off to the side
    // A point only inside the grown live region is now hit-testable.
    expect(hitTestLayer(layer, doc([layer]), { x: 45, y: 45 }, liveRect)).toBe(true);
    // The original persisted region is still hit-testable.
    expect(hitTestLayer(layer, doc([layer]), { x: 5, y: 5 }, liveRect)).toBe(true);
  });
});

describe('a layer erased to nothing behaves exactly like a brand-new one', () => {
  /**
   * The reported bug: an erased layer kept a full-size transparent bitmap, so it still
   * drew a movable, transformable outline around nothing. The paint-cache trim now
   * clears it to `{ bitmap: null }` — a freshly-added layer's source — so every
   * affordance derived from `hittableLayerRect` disappears. Pinning that equivalence
   * is why the fix needs no change to the overlay or the tools.
   */
  it('reports no hittable extent once its bitmap has been cleared', () => {
    const erased = paintLayer('erased');
    const fresh = paintLayer('fresh');

    expect(hittableLayerSize(erased, doc([erased]))).toBeNull();
    expect(hittableLayerSize(erased, doc([erased]))).toEqual(hittableLayerSize(fresh, doc([fresh])));
  });

  it('draws no move outline, even under a non-identity transform', () => {
    const erased: CanvasLayerContract = {
      ...paintLayer('erased'),
      transform: { rotation: 0.6, scaleX: 2.5, scaleY: 1.75, x: 40, y: 25 },
    };
    expect(layerOutlineCorners(erased, doc([erased]))).toBeNull();
    expect(layerOutlineCorners(erased, doc([erased]), { x: 10, y: 10 })).toBeNull();
  });

  it('cannot be grabbed anywhere it used to cover', () => {
    const erased = paintLayer('erased');
    for (const point of [
      { x: 0, y: 0 },
      { x: 20, y: 20 },
      { x: 60, y: 40 },
    ]) {
      expect(hitTestLayer(erased, doc([erased]), point)).toBe(false);
    }
  });

  it('applies to a cleared MASK too, since masks share the paint path', () => {
    const erased = maskLayer('erased-mask');
    expect(hittableLayerSize(erased, doc([erased]))).toBeNull();
    expect(layerOutlineCorners(erased, doc([erased]))).toBeNull();
  });
});

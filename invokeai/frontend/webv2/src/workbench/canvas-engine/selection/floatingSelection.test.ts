import type { PlacedSurface, Rect } from '@workbench/canvas-engine/types';

import { fromTRS, identity } from '@workbench/canvas-engine/math/mat2d';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it } from 'vitest';

import {
  documentDeltaToLocal,
  floatDocumentMatrix,
  liftSelectedPixels,
  transformPlacedMask,
} from './floatingSelection';

const backend = createTestStubRasterBackend();

const placed = (rect: Rect): PlacedSurface => ({
  rect,
  surface: backend.createSurface(Math.max(0, rect.width), Math.max(0, rect.height)),
});

/** A layer matrix from the usual transform fields. */
const matrixOf = (t: { x?: number; y?: number; rotation?: number; scaleX?: number; scaleY?: number }) =>
  fromTRS({ x: t.x ?? 0, y: t.y ?? 0 }, t.rotation ?? 0, t.scaleX ?? 1, t.scaleY ?? t.scaleX ?? 1);

/** Rounds a matrix's components so exact trig comparisons stay readable. */
const rounded = (value: number): number => Math.round(value * 1e6) / 1e6;

describe('documentDeltaToLocal', () => {
  it('passes a delta through an identity layer unchanged', () => {
    expect(documentDeltaToLocal(identity(), { x: 10, y: -4 })).toEqual({ x: 10, y: -4 });
  });

  it('ignores the layer translation — a delta is a vector, not a point', () => {
    expect(documentDeltaToLocal(matrixOf({ x: 500, y: -300 }), { x: 10, y: -4 })).toEqual({ x: 10, y: -4 });
  });

  it('divides by the layer scale', () => {
    const local = documentDeltaToLocal(matrixOf({ scaleX: 2 }), { x: 10, y: 8 });
    expect(rounded(local.x)).toBe(5);
    expect(rounded(local.y)).toBe(4);
  });

  it('un-rotates by the layer rotation', () => {
    // The layer is rotated a quarter turn, so dragging right in document space
    // must move the pixels "up" in the layer's own frame.
    const local = documentDeltaToLocal(matrixOf({ rotation: Math.PI / 2 }), { x: 10, y: 0 });
    expect(rounded(local.x)).toBe(0);
    expect(rounded(local.y)).toBe(-10);
  });
});

describe('floatDocumentMatrix', () => {
  it('is the float transform itself under an identity layer', () => {
    const float = matrixOf({ x: 7, y: -3 });
    expect(floatDocumentMatrix(identity(), float)).toEqual(float);
  });

  it('leaves a translate unchanged under a translated layer (translations commute)', () => {
    const result = floatDocumentMatrix(matrixOf({ x: 100, y: 40 }), matrixOf({ x: 7, y: -3 }));
    expect(rounded(result!.e)).toBe(7);
    expect(rounded(result!.f)).toBe(-3);
  });

  it('scales a local translate up by the layer scale', () => {
    // 5 layer-local px on a 2× layer is 10 document px.
    const result = floatDocumentMatrix(matrixOf({ scaleX: 2 }), matrixOf({ x: 5, y: 0 }));
    expect(rounded(result!.e)).toBe(10);
    expect(rounded(result!.f)).toBe(0);
  });

  it('rotates a local translate into the document frame', () => {
    const result = floatDocumentMatrix(matrixOf({ rotation: Math.PI / 2 }), matrixOf({ x: 10, y: 0 }));
    expect(rounded(result!.e)).toBe(0);
    expect(rounded(result!.f)).toBe(10);
  });
});

describe('liftSelectedPixels', () => {
  const cache = () => placed({ height: 100, width: 100, x: 0, y: 0 });

  it('cuts the selection ∩ content region under an identity layer', () => {
    const lifted = liftSelectedPixels({
      backend,
      cache: cache(),
      layerMatrix: identity(),
      mask: placed({ height: 30, width: 30, x: 20, y: 20 }),
    });
    expect(lifted?.pixels.rect).toEqual({ height: 30, width: 30, x: 20, y: 20 });
    // The stencil covers exactly the same region, so the cut and the copy agree.
    expect(lifted?.localMask.rect).toEqual(lifted?.pixels.rect);
  });

  it('clips the region to what the layer actually holds', () => {
    const lifted = liftSelectedPixels({
      backend,
      cache: cache(),
      layerMatrix: identity(),
      mask: placed({ height: 50, width: 50, x: 80, y: 80 }),
    });
    // The selection runs off the layer's right/bottom edge.
    expect(lifted?.pixels.rect).toEqual({ height: 20, width: 20, x: 80, y: 80 });
  });

  it('maps the document-space mask into layer-local space through the layer translation', () => {
    const lifted = liftSelectedPixels({
      backend,
      cache: cache(),
      layerMatrix: matrixOf({ x: 10, y: 5 }),
      mask: placed({ height: 30, width: 30, x: 20, y: 20 }),
    });
    // Document (20,20) sits at layer-local (10,15) once the layer's own offset is undone.
    expect(lifted?.pixels.rect).toEqual({ height: 30, width: 30, x: 10, y: 15 });
  });

  it('maps through the layer scale', () => {
    const lifted = liftSelectedPixels({
      backend,
      cache: cache(),
      layerMatrix: matrixOf({ scaleX: 2 }),
      mask: placed({ height: 40, width: 40, x: 20, y: 20 }),
    });
    expect(lifted?.pixels.rect).toEqual({ height: 20, width: 20, x: 10, y: 10 });
  });

  it('maps through a layer rotation', () => {
    const lifted = liftSelectedPixels({
      backend,
      cache: placed({ height: 200, width: 200, x: -100, y: -100 }),
      layerMatrix: matrixOf({ rotation: Math.PI / 2 }),
      mask: placed({ height: 20, width: 10, x: 30, y: 40 }),
    });
    // A quarter turn swaps the axes: document x becomes local -y and vice versa,
    // so document [30,40]×[40,60] lands on local [40,60]×[-40,-30]. The region is
    // rounded OUTWARD, so trig landing a hair off an integer may widen it by a
    // pixel — harmless, since the stencil's alpha is zero out there.
    const rect = lifted!.pixels.rect;
    expect(rect.x).toBe(40);
    expect(rect.width).toBe(20);
    expect(rect.y).toBeGreaterThanOrEqual(-41);
    expect(rect.y).toBeLessThanOrEqual(-40);
    expect(rect.y + rect.height).toBeGreaterThanOrEqual(-30);
    expect(rect.y + rect.height).toBeLessThanOrEqual(-29);
  });

  it('returns null when the selection misses the layer entirely', () => {
    expect(
      liftSelectedPixels({
        backend,
        cache: cache(),
        layerMatrix: identity(),
        mask: placed({ height: 10, width: 10, x: 500, y: 500 }),
      })
    ).toBeNull();
  });

  it('returns null for an empty cache', () => {
    expect(
      liftSelectedPixels({
        backend,
        cache: placed({ height: 0, width: 0, x: 0, y: 0 }),
        layerMatrix: identity(),
        mask: placed({ height: 30, width: 30, x: 20, y: 20 }),
      })
    ).toBeNull();
  });

  it('returns null for a non-invertible layer matrix (a collapsed scale)', () => {
    expect(
      liftSelectedPixels({
        backend,
        cache: cache(),
        layerMatrix: matrixOf({ scaleX: 0 }),
        mask: placed({ height: 30, width: 30, x: 20, y: 20 }),
      })
    ).toBeNull();
  });
});

describe('transformPlacedMask', () => {
  it('re-places the mask rect through the matrix', () => {
    const moved = transformPlacedMask(
      backend,
      placed({ height: 30, width: 30, x: 20, y: 20 }),
      matrixOf({ x: 5, y: 7 })
    );
    expect(moved?.rect).toEqual({ height: 30, width: 30, x: 25, y: 27 });
  });

  it('grows the rect outward for a scale', () => {
    const moved = transformPlacedMask(
      backend,
      placed({ height: 30, width: 30, x: 20, y: 20 }),
      matrixOf({ scaleX: 2 })
    );
    expect(moved?.rect).toEqual({ height: 60, width: 60, x: 40, y: 40 });
  });

  it('returns null when the transform collapses the mask', () => {
    expect(
      transformPlacedMask(backend, placed({ height: 30, width: 30, x: 20, y: 20 }), matrixOf({ scaleX: 0 }))
    ).toBeNull();
  });
});

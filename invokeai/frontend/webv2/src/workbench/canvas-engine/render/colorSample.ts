/**
 * Samples composited layer pixels into a translated scratch (1x1 for a pick, a small square for the picker's loupe)
 * using normal ordering and display effects. Excludes background/checkerboard and staged previews so uncovered points
 * return null.
 */

import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { Mat2d, Vec2 } from '@workbench/canvas-engine/types';

import type { LayerCacheStore } from './layerCache';
import type { RasterBackend, RasterSurface } from './raster';

import { compositeDocument, reusePreparation, type CompositeOptions, type CompositePreparation } from './compositor';

/** An RGBA sample, channels in `[0, 255]`. */
export interface RgbaSample {
  r: number;
  g: number;
  b: number;
  a: number;
}

export type ColorSampleProviders = Pick<CompositeOptions, 'adjustedSurface' | 'derivedSurfaces' | 'groupSurface'>;

const RAW_PIXELS: ColorSampleProviders = {};

/** A square of composited document pixels centered on a sampled pixel. */
export interface ColorSampleArea {
  /** `size`×`size` pixels; the sampled pixel is the center one. Reused by the next area sample. */
  readonly pixels: RasterSurface;
  readonly center: RgbaSample | null;
}

export interface ColorSampler {
  /**
   * Floors document coordinates to a pixel; transformed content remains pickable outside document dimensions.
   * Returns null for zero-alpha coverage. Repeating a pixel of an unchanged document and cache revision reuses the
   * previous result.
   */
  sample(
    doc: CanvasDocumentContractV3,
    layers: LayerCacheStore,
    docPoint: Vec2,
    providers?: ColorSampleProviders
  ): RgbaSample | null;
  /** The pixels around `docPoint` (odd `size`), with the same coverage rules and reuse as {@link sample}. */
  sampleArea(
    doc: CanvasDocumentContractV3,
    layers: LayerCacheStore,
    docPoint: Vec2,
    size: number,
    providers?: ColorSampleProviders
  ): ColorSampleArea | null;
}

const sampleAt = (data: Uint8ClampedArray, offset: number): RgbaSample | null => {
  const alpha = data[offset + 3] ?? 0;
  return alpha === 0 ? null : { a: alpha, b: data[offset + 2] ?? 0, g: data[offset + 1] ?? 0, r: data[offset] ?? 0 };
};

/** A sampler that keeps its scratch, draw plan and last sample across a picking gesture. */
export const createColorSampler = (backend: RasterBackend): ColorSampler => {
  let preparation: CompositePreparation | null = null;
  const scratches = new Map<number, RasterSurface>();
  interface Last {
    doc: CanvasDocumentContractV3;
    layers: LayerCacheStore;
    revision: number;
    providers: ColorSampleProviders;
    px: number;
    py: number;
    area: ColorSampleArea;
  }
  const lastBySize = new Map<number, Last>();

  /** Composites the `size`-square around a pixel; the canonical compositor keeps placement and display effects. */
  const composite = (
    doc: CanvasDocumentContractV3,
    layers: LayerCacheStore,
    docPoint: Vec2,
    size: number,
    providers: ColorSampleProviders
  ): ColorSampleArea | null => {
    const px = Math.floor(docPoint.x);
    const py = Math.floor(docPoint.y);
    if (!Number.isFinite(px) || !Number.isFinite(py)) {
      return null;
    }
    const revision = layers.revision();
    const last = lastBySize.get(size);
    if (
      last?.doc === doc &&
      last.layers === layers &&
      last.revision === revision &&
      last.providers === providers &&
      last.px === px &&
      last.py === py
    ) {
      return last.area;
    }

    let scratch = scratches.get(size);
    if (!scratch) {
      scratch = backend.createSurface(size, size);
      scratches.set(size, scratch);
    }
    const half = (size - 1) / 2;
    preparation = reusePreparation(preparation, doc, { backend, ...providers });
    const view: Mat2d = { a: 1, b: 0, c: 0, d: 1, e: half - px, f: half - py };
    // Omitting checkerboard and staged previews keeps empty space transparent.
    compositeDocument(scratch, doc, layers, view, { backend, ...providers, preparation });

    const { data } = scratch.ctx.getImageData(half, half, 1, 1);
    const area: ColorSampleArea = { center: sampleAt(data, 0), pixels: scratch };
    lastBySize.set(size, { area, doc, layers, providers, px, py, revision });
    return area;
  };

  return {
    sample: (doc, layers, docPoint, providers = RAW_PIXELS) =>
      composite(doc, layers, docPoint, 1, providers)?.center ?? null,
    sampleArea: (doc, layers, docPoint, size, providers = RAW_PIXELS) =>
      composite(doc, layers, docPoint, Math.max(1, Math.floor(size / 2) * 2 + 1), providers),
  };
};

/**
 * Samples composited layer pixels into a translated 1x1 scratch using normal ordering and display effects.
 * Excludes background/checkerboard and staged previews so uncovered points return null.
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
}

/** A sampler that keeps its scratch, draw plan and last sample across a picking gesture. */
export const createColorSampler = (backend: RasterBackend): ColorSampler => {
  let scratch: RasterSurface | null = null;
  let preparation: CompositePreparation | null = null;
  let last: {
    doc: CanvasDocumentContractV3;
    layers: LayerCacheStore;
    revision: number;
    providers: ColorSampleProviders;
    px: number;
    py: number;
    result: RgbaSample | null;
  } | null = null;

  return {
    sample: (doc, layers, docPoint, providers = RAW_PIXELS) => {
      const px = Math.floor(docPoint.x);
      const py = Math.floor(docPoint.y);
      if (!Number.isFinite(px) || !Number.isFinite(py)) {
        return null;
      }
      const revision = layers.revision();
      if (
        last?.doc === doc &&
        last.layers === layers &&
        last.revision === revision &&
        last.providers === providers &&
        last.px === px &&
        last.py === py
      ) {
        return last.result;
      }

      scratch ??= backend.createSurface(1, 1);
      preparation = reusePreparation(preparation, doc, { backend, ...providers });
      const view: Mat2d = { a: 1, b: 0, c: 0, d: 1, e: -px, f: -py };
      // Use canonical compositing for placement and display effects, omitting checkerboard/staged previews to
      // retain transparent empty space.
      compositeDocument(scratch, doc, layers, view, { backend, ...providers, preparation });

      const { data } = scratch.ctx.getImageData(0, 0, 1, 1);
      const alpha = data[3] ?? 0;
      const result = alpha === 0 ? null : { a: alpha, b: data[2] ?? 0, g: data[1] ?? 0, r: data[0] ?? 0 };
      last = { doc, layers, providers, px, py, result, revision };
      return result;
    },
  };
};

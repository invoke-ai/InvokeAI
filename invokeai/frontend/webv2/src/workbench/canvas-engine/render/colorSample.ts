/**
 * Samples composited layer pixels into a translated scratch (1x1 for a pick, a small square for the picker's loupe)
 * using normal ordering and display effects. Excludes background/checkerboard and staged previews so uncovered points
 * return null.
 */

import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { Mat2d, Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint, invert } from '@workbench/canvas-engine/math/mat2d';

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

/** The viewport's patterned-mask drawing; the hatch is laid out in screen space, so samples need its phase. */
export interface ColorSampleMaskPatterns {
  readonly tile: NonNullable<CompositeOptions['maskPatternTile']>;
  /** The viewport's current backing scale, document-to-device matrix and smoothing. */
  readonly screen: () => { dpr: number; view: Mat2d; smoothing: boolean };
}

export type ColorSampleProviders = Pick<
  CompositeOptions,
  'adjustedSurface' | 'derivedSurfaces' | 'groupSurface' | 'showOverlayStacks'
> & {
  /** Absent draws patterned masks solid. */
  maskPatterns?: ColorSampleMaskPatterns;
};

const RAW_PIXELS: ColorSampleProviders = {};

/** A square of composited document pixels centered on a sampled pixel. */
export interface ColorSampleArea {
  /** `size`×`size` pixels; the sampled pixel is the center one. Reused by the next area sample. */
  readonly pixels: RasterSurface;
  readonly center: RgbaSample | null;
}

export interface ColorSampler {
  /**
   * Floors document coordinates to a pixel; transformed content remains pickable outside document dimensions. Returns
   * null for zero-alpha coverage. Repeating a pixel of an unchanged document and cache revision reuses the previous
   * result. With `maskPatterns`, `screenPoint` (viewport CSS pixels) picks the device pixel instead, and patterned
   * masks read as the viewport draws them there.
   */
  sample(
    doc: CanvasDocumentContractV3,
    layers: LayerCacheStore,
    docPoint: Vec2,
    providers?: ColorSampleProviders,
    screenPoint?: Vec2
  ): RgbaSample | null;
  /** The pixels around `docPoint` (odd `size`), with the same coverage rules and reuse as {@link sample}. */
  sampleArea(
    doc: CanvasDocumentContractV3,
    layers: LayerCacheStore,
    docPoint: Vec2,
    size: number,
    providers?: ColorSampleProviders,
    screenPoint?: Vec2
  ): ColorSampleArea | null;
}

const sampleAt = (data: Uint8ClampedArray, offset: number): RgbaSample | null => {
  const alpha = data[offset + 3] ?? 0;
  return alpha === 0 ? null : { a: alpha, b: data[offset + 2] ?? 0, g: data[offset + 1] ?? 0, r: data[offset] ?? 0 };
};

/** How a sample scratch lines up with the viewport's device pixels. */
interface DevicePlacement {
  dpr: number;
  view: Mat2d;
  smoothing: boolean;
  /** Places the scratch's center pixel center on the center of the device pixel under the pointer. */
  fromTarget: Mat2d;
  /** That device pixel's center in document space. */
  docPoint: Vec2;
}

/**
 * The device pixel comes from the pointer's own screen position: a round trip through document space can land on its
 * neighbor. Scratch neighbors step one document pixel, so a loupe cell shows the screen where that pixel is drawn.
 */
const placeOnDevice = (
  { dpr, smoothing, view }: ReturnType<ColorSampleMaskPatterns['screen']>,
  screenPoint: Vec2,
  half: number
): DevicePlacement | null => {
  const toDocument = invert(view);
  if (!toDocument) {
    return null;
  }
  const pixel = { x: Math.floor(screenPoint.x * dpr) + 0.5, y: Math.floor(screenPoint.y * dpr) + 0.5 };
  const linear = { ...view, e: 0, f: 0 };
  const center = applyToPoint(linear, { x: half + 0.5, y: half + 0.5 });
  return {
    docPoint: applyToPoint(toDocument, pixel),
    dpr,
    fromTarget: { ...linear, e: pixel.x - center.x, f: pixel.y - center.y },
    smoothing,
    view,
  };
};

const sameMatrix = (a: Mat2d, b: Mat2d): boolean =>
  a.a === b.a && a.b === b.b && a.c === b.c && a.d === b.d && a.e === b.e && a.f === b.f;

const samePlacement = (a: DevicePlacement | null, b: DevicePlacement | null): boolean =>
  a === b ||
  (!!a &&
    !!b &&
    a.dpr === b.dpr &&
    a.smoothing === b.smoothing &&
    sameMatrix(a.view, b.view) &&
    sameMatrix(a.fromTarget, b.fromTarget));

/** A sampler that keeps its scratch, draw plan and last sample across a picking gesture. */
export const createColorSampler = (backend: RasterBackend): ColorSampler => {
  let preparation: CompositePreparation | null = null;
  const scratches = new Map<number, RasterSurface>();
  // Grows to the largest area sampled; patterned masks stencil through it.
  let stencil: RasterSurface | null = null;
  const getStencil = (width: number, height: number): RasterSurface => {
    if (!stencil || stencil.width < width || stencil.height < height) {
      stencil = backend.createSurface(Math.max(width, stencil?.width ?? 0), Math.max(height, stencil?.height ?? 0));
    }
    return stencil;
  };
  interface Last {
    doc: CanvasDocumentContractV3;
    layers: LayerCacheStore;
    revision: number;
    providers: ColorSampleProviders;
    px: number;
    py: number;
    placement: DevicePlacement | null;
    area: ColorSampleArea;
  }
  const lastBySize = new Map<number, Last>();

  /** Composites the `size`-square around a pixel; the canonical compositor keeps placement and display effects. */
  const composite = (
    doc: CanvasDocumentContractV3,
    layers: LayerCacheStore,
    docPoint: Vec2,
    size: number,
    providers: ColorSampleProviders,
    screenPoint: Vec2 | undefined
  ): ColorSampleArea | null => {
    const half = (size - 1) / 2;
    const { maskPatterns, ...compositeProviders } = providers;
    const placement =
      maskPatterns && screenPoint && Number.isFinite(screenPoint.x) && Number.isFinite(screenPoint.y)
        ? placeOnDevice(maskPatterns.screen(), screenPoint, half)
        : null;
    const point = placement?.docPoint ?? docPoint;
    const px = Math.floor(point.x);
    const py = Math.floor(point.y);
    if (!Number.isFinite(px) || !Number.isFinite(py)) {
      return null;
    }
    const view: Mat2d = { a: 1, b: 0, c: 0, d: 1, e: half - px, f: half - py };
    const revision = layers.revision();
    const last = lastBySize.get(size);
    if (
      last?.doc === doc &&
      last.layers === layers &&
      last.revision === revision &&
      last.providers === providers &&
      last.px === px &&
      last.py === py &&
      samePlacement(last.placement, placement)
    ) {
      return last.area;
    }

    let scratch = scratches.get(size);
    if (!scratch) {
      scratch = backend.createSurface(size, size);
      scratches.set(size, scratch);
    }
    preparation = reusePreparation(preparation, doc, { backend, ...compositeProviders });
    // Omitting checkerboard and staged previews keeps empty space transparent.
    compositeDocument(scratch, doc, layers, view, {
      backend,
      ...compositeProviders,
      ...(maskPatterns &&
        placement && {
          maskPatternTile: maskPatterns.tile,
          screenMask: {
            device: { fromTarget: placement.fromTarget, smoothing: placement.smoothing, view: placement.view },
            dpr: placement.dpr,
            getScratch: getStencil,
          },
        }),
      preparation,
    });

    const { data } = scratch.ctx.getImageData(half, half, 1, 1);
    const area: ColorSampleArea = { center: sampleAt(data, 0), pixels: scratch };
    lastBySize.set(size, { area, doc, layers, providers, px, py, revision, placement });
    return area;
  };

  return {
    sample: (doc, layers, docPoint, providers = RAW_PIXELS, screenPoint) =>
      composite(doc, layers, docPoint, 1, providers, screenPoint)?.center ?? null,
    sampleArea: (doc, layers, docPoint, size, providers = RAW_PIXELS, screenPoint) =>
      composite(doc, layers, docPoint, Math.max(1, Math.floor(size / 2) * 2 + 1), providers, screenPoint),
  };
};

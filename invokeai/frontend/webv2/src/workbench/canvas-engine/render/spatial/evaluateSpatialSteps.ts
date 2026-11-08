// SPDX-License-Identifier: Apache-2.0
/**
 * Canvas evaluation of the backend's Clarity, Texture and Contrast Dehaze nodes (`invokeai/backend/image_util/`).
 * Each step equals the node run on the layer cropped to its alpha bounds: transparent padding never changes the
 * result, and pixels outside those bounds, hidden (alpha 0) pixels and alpha itself are never written.
 */

import type { Rect } from '@workbench/canvas-engine/types';

import { alphaBounds } from '@workbench/canvas-engine/render/alphaBounds';
import { propagateNearestVisibleLabels } from '@workbench/canvas-engine/render/spatial/alphaFill';
import { boxMean, edgeAwareBand, type FilterScratch } from '@workbench/canvas-engine/render/spatial/guidedFilter';
import {
  clipToSrgbGamut,
  linearSrgbFromOklab,
  oklabFromSrgb8,
  roundHalfEven,
  srgb8FromLinear,
} from '@workbench/canvas-engine/render/spatial/numerics';
import {
  MAX_SPATIAL_PIXELS,
  SPATIAL_SCRATCH_BYTES_PER_PIXEL,
  type SpatialAdjustmentStep,
  type SpatialEvaluation,
} from '@workbench/canvas-engine/render/spatial/spatialSteps';

const CLARITY_EPSILON = 0.002;
const TEXTURE_EPSILON = 0.0008;
const DEHAZE_EPSILON = 0.002;

const OK: SpatialEvaluation = { status: 'ok' };
const TOO_LARGE: SpatialEvaluation = { status: 'too-large' };

const clamp = (value: number, min: number, max: number): number => (value < min ? min : value > max ? max : value);

/** Clamps a step into the node's accepted ranges; null when it cannot change pixels. */
const resolveStep = (step: SpatialAdjustmentStep): SpatialAdjustmentStep | null => {
  if (!Number.isFinite(step.amount) || step.amount === 0) {
    return null;
  }
  const amount = clamp(step.amount, -200, 200);
  switch (step.kind) {
    case 'clarity':
      return Number.isFinite(step.size) ? { amount, kind: 'clarity', size: clamp(step.size, 4, 1000) } : null;
    case 'texture':
      return Number.isFinite(step.size) ? { amount, kind: 'texture', size: clamp(step.size, 1, 100) } : null;
    case 'dehaze':
      return { amount, kind: 'dehaze' };
  }
};

/** One evaluation's view of the layer: its alpha bounds and the bounds-sized planes every step reuses. */
type Workspace = {
  readonly pixels: Uint8ClampedArray;
  readonly stride: number;
  readonly bounds: Rect;
  /** Visible pixels label themselves; hidden ones name the visible pixel whose lightness they borrow. Null when
   * nothing inside the bounds is hidden. */
  readonly labels: Int32Array | null;
  readonly lightness: Float32Array;
  /** The step's band, then its adjusted lightness. */
  readonly adjusted: Float32Array;
  readonly scratch: FilterScratch;
  readonly color: Float64Array;
};

const LABEL_BYTES_PER_PIXEL = 4;

const hasHiddenPixels = (pixels: Uint8ClampedArray, stride: number, bounds: Rect): boolean => {
  for (let y = 0; y < bounds.height; y++) {
    const alphaOffset = ((bounds.y + y) * stride + bounds.x) * 4 + 3;
    for (let x = 0; x < bounds.width; x++) {
      if (pixels[alphaOffset + x * 4] === 0) {
        return true;
      }
    }
  }
  return false;
};

/** Visible pixels label themselves; hidden pixels get the visible pixel the backend's fill copies from. */
const fillLabels = (
  pixels: Uint8ClampedArray,
  stride: number,
  bounds: Rect,
  labels: Int32Array,
  distances: Uint32Array
) => {
  for (let y = 0; y < bounds.height; y++) {
    const rowOffset = y * bounds.width;
    const alphaOffset = ((bounds.y + y) * stride + bounds.x) * 4 + 3;
    for (let x = 0; x < bounds.width; x++) {
      labels[rowOffset + x] = pixels[alphaOffset + x * 4] === 0 ? -1 : rowOffset + x;
    }
  }
  propagateNearestVisibleLabels(labels, bounds.width, bounds.height, distances);
};

/**
 * Every per-pixel plane comes from one buffer sized by {@link SPATIAL_SCRATCH_BYTES_PER_PIXEL}, so the service's
 * memory reservation and the actual allocation cannot drift apart.
 */
const createWorkspace = (pixels: Uint8ClampedArray, stride: number, bounds: Rect): Workspace => {
  const count = bounds.width * bounds.height;
  const hidden = hasHiddenPixels(pixels, stride, bounds);
  const bytesPerPixel = SPATIAL_SCRATCH_BYTES_PER_PIXEL - (hidden ? 0 : LABEL_BYTES_PER_PIXEL);
  const buffer = new ArrayBuffer(count * bytesPerPixel);
  let offset = 0;
  const plane = (): Float32Array => {
    const view = new Float32Array(buffer, offset, count);
    offset += view.byteLength;
    return view;
  };
  const scratch: FilterScratch = {
    columnSums: new Float64Array(bounds.width),
    first: plane(),
    second: plane(),
    third: plane(),
  };
  const lightness = plane();
  const adjusted = plane();
  let labels: Int32Array | null = null;
  if (hidden) {
    labels = new Int32Array(buffer, offset, count);
    offset += labels.byteLength;
    // The distances are dead before any filter runs, so they borrow a filter plane's memory.
    fillLabels(pixels, stride, bounds, labels, new Uint32Array(buffer, scratch.first.byteOffset, count));
  }
  if (offset !== buffer.byteLength) {
    throw new Error('Spatial scratch layout does not match SPATIAL_SCRATCH_BYTES_PER_PIXEL');
  }
  return { adjusted, bounds, color: new Float64Array(3), labels, lightness, pixels, scratch, stride };
};

/** Oklab lightness of the current pixels; hidden pixels take their labelled visible pixel's lightness. */
const loadLightness = ({ bounds, color, labels, lightness, pixels, stride }: Workspace): void => {
  for (let y = 0; y < bounds.height; y++) {
    const rowOffset = y * bounds.width;
    const pixelOffset = ((bounds.y + y) * stride + bounds.x) * 4;
    for (let x = 0; x < bounds.width; x++) {
      const p = pixelOffset + x * 4;
      if (pixels[p + 3] !== 0) {
        oklabFromSrgb8(pixels[p], pixels[p + 1], pixels[p + 2], color);
        lightness[rowOffset + x] = color[0];
      }
    }
  }
  if (labels) {
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] !== i) {
        lightness[i] = lightness[labels[i]];
      }
    }
  }
};

/** Writes visible pixels back with `adjusted` lightness and their own chroma scaled by float32 `chromaScale`. */
const storeAdjusted = ({ adjusted, bounds, color, pixels, stride }: Workspace, chromaScale: number): void => {
  for (let y = 0; y < bounds.height; y++) {
    const rowOffset = y * bounds.width;
    const pixelOffset = ((bounds.y + y) * stride + bounds.x) * 4;
    for (let x = 0; x < bounds.width; x++) {
      const p = pixelOffset + x * 4;
      if (pixels[p + 3] === 0) {
        continue;
      }
      oklabFromSrgb8(pixels[p], pixels[p + 1], pixels[p + 2], color);
      // The backend stores Oklab as float32 and scales chroma in float32.
      const a = Math.fround(Math.fround(color[1]) * chromaScale);
      const b = Math.fround(Math.fround(color[2]) * chromaScale);
      linearSrgbFromOklab(clamp(adjusted[rowOffset + x], 0, 1), a, b, color);
      clipToSrgbGamut(color);
      pixels[p] = srgb8FromLinear(color[0]);
      pixels[p + 1] = srgb8FromLinear(color[1]);
      pixels[p + 2] = srgb8FromLinear(color[2]);
    }
  }
};

/** Inner and outer band radii shared by Clarity and Texture. */
const bandRadii = (size: number): readonly [number, number] => {
  const inner = Math.max(1, roundHalfEven(size / 4));
  return [inner, Math.max(inner + 1, roundHalfEven(size))];
};

const applyClarity = (workspace: Workspace, amount: number, size: number): void => {
  const { adjusted, bounds, lightness, scratch } = workspace;
  const [inner, outer] = bandRadii(size);
  edgeAwareBand(lightness, adjusted, bounds.width, bounds.height, inner, outer, CLARITY_EPSILON, scratch);
  const strength = amount / 100;
  for (let i = 0; i < adjusted.length; i++) {
    const value = lightness[i];
    const clipped = clamp(value, 0, 1);
    adjusted[i] = value + strength * (4 * clipped * (1 - clipped)) * adjusted[i];
  }
  storeAdjusted(workspace, 1);
};

const applyTexture = (workspace: Workspace, amount: number, size: number): void => {
  const { adjusted, bounds, lightness, scratch } = workspace;
  const [inner, outer] = bandRadii(size);
  edgeAwareBand(lightness, adjusted, bounds.width, bounds.height, inner, outer, TEXTURE_EPSILON, scratch);
  if (amount > 0) {
    // Fade sharpening out where the local band RMS is noise-level flat.
    const meanSquare = scratch.first;
    const rmsRadius = Math.max(2, roundHalfEven(size / 3));
    boxMean(adjusted, meanSquare, bounds.width, bounds.height, rmsRadius, true, scratch.columnSums);
    for (let i = 0; i < adjusted.length; i++) {
      const weight = clamp((Math.sqrt(Math.max(meanSquare[i], 0)) - 0.002) / (0.01 - 0.002), 0, 1);
      adjusted[i] *= weight * weight * (3 - 2 * weight);
    }
  }
  const strength = amount / 100;
  for (let i = 0; i < adjusted.length; i++) {
    adjusted[i] = lightness[i] + strength * adjusted[i];
  }
  storeAdjusted(workspace, 1);
};

const applyDehaze = (workspace: Workspace, amount: number): void => {
  const { adjusted, bounds, lightness, scratch } = workspace;
  const radius = Math.max(2, Math.min(1000, roundHalfEven(Math.min(bounds.width, bounds.height) / 8)));
  edgeAwareBand(lightness, adjusted, bounds.width, bounds.height, 0, radius, DEHAZE_EPSILON, scratch);
  const strength = amount / 100;
  const contrast = 1 + 0.2 * strength;
  for (let i = 0; i < adjusted.length; i++) {
    adjusted[i] = 0.5 + contrast * (lightness[i] - 0.5) + strength * adjusted[i];
  }
  storeAdjusted(workspace, Math.fround(1 + 0.2 * strength));
};

/** Applies `steps` in order to unpremultiplied RGBA8 `pixels` (row-major, width*height*4) in place. */
export const evaluateSpatialSteps = (
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  steps: readonly SpatialAdjustmentStep[]
): SpatialEvaluation => {
  const effective: SpatialAdjustmentStep[] = [];
  for (const step of steps) {
    const resolved = resolveStep(step);
    if (resolved) {
      effective.push(resolved);
    }
  }
  if (effective.length === 0) {
    return OK;
  }
  const bounds = alphaBounds({ data: pixels, height, width });
  if (bounds.width === 0 || bounds.height === 0) {
    return OK;
  }
  if (bounds.width * bounds.height > MAX_SPATIAL_PIXELS) {
    return TOO_LARGE;
  }
  const workspace = createWorkspace(pixels, width, bounds);
  for (const step of effective) {
    loadLightness(workspace);
    switch (step.kind) {
      case 'clarity':
        applyClarity(workspace, step.amount, step.size);
        break;
      case 'texture':
        applyTexture(workspace, step.amount, step.size);
        break;
      case 'dehaze':
        applyDehaze(workspace, step.amount);
        break;
    }
  }
  return OK;
};

// SPDX-License-Identifier: Apache-2.0
/**
 * Turns a fill level map into the fill's coverage for one set of region parameters.
 *
 * 1. Threshold: the region is `level ≤ round(tolerance × 2.55)`.
 * 2. Close gaps `g` (contiguous only): a gap in the barrier (`level` above the threshold) at most `g` px wide closes,
 *    odd widths included. The barrier grows by `g / 2` px measured on a half-pixel grid (a pixel stays open when any
 *    of its four half-pixel samples is more than `g / 2` px from every barrier half-pixel), and the seed refloods the
 *    open pixels. A grown-over pixel of the thresholded region then joins the fill when the flood is its nearest open
 *    pixel and it is reached from the flood by 8-connected steps over grown-over pixels that strictly descend in
 *    distance to the barrier. So corners fill to the line, a closed gap fills to its middle, and nothing passes a
 *    gap into the space behind it. A seed whose pixel the grown barrier covers leaves the region unclosed.
 * 3. Expand `e`: the region's boundary sits half a pixel outside its outermost pixel centers; a pixel belongs to the
 *    expanded region when its center lies inside the boundary moved outwards by `e` (inwards for negative `e`).
 * 4. Coverage: hard is 0 or 255. Anti-aliased coverage ramps linearly across the boundary over two pixels, so a
 *    straight edge keeps its area and only the boundary's 8-connected pixel ring gets intermediate values.
 *
 * Distances are exact Euclidean (Felzenszwalb and Huttenlocher's separable transform), so expansion is round and
 * the anti-alias ramp follows real distances; it costs about the same linear passes as a 5-7-11 chamfer.
 * All passes run over the region's bounds plus the reach of the expansion, never the whole sample, and the sample
 * edge is neither inside nor outside: it does not erode, shrink or anti-alias the region.
 */

import type { FillLevelMap } from '@workbench/canvas-engine/render/fill/fillLevelMap';
import type { Rect } from '@workbench/canvas-engine/types';

import { FILL_LEVELS } from '@workbench/canvas-engine/render/fill/fillLevelMap';

export interface FillRegionParams {
  /** 0..100; 100 takes the whole sample (contiguous: everything connected to the seed, which is everything). */
  readonly tolerance: number;
  /** Pixels; 0 disables gap closing. */
  readonly closeGaps: number;
  /** Pixels; negative shrinks. */
  readonly expand: number;
  readonly antiAlias: boolean;
}

/** `coverage` holds `bounds.width * bounds.height` bytes (0..255) for sample pixels inside `bounds`. */
export type FillRegion =
  | { readonly bounds: Rect; readonly coverage: Uint8Array; readonly pixelCount: number }
  | { readonly bounds: null; readonly coverage: null; readonly pixelCount: 0 };

export const EMPTY_FILL_REGION: FillRegion = { bounds: null, coverage: null, pixelCount: 0 };

/** Squared distances cap here: exact below (1024 px), far beyond any radius the passes threshold at. */
const DISTANCE_CAP = 1 << 20;
/** Half-pixel squared distances are stored as 16-bit: exact up to 255 half-pixels, beyond any gap radius (16 px). */
const HALF_PIXEL_CAP = 0xffff;

const OUTSIDE = 0;
const INSIDE = 1;
const OPEN = 2;
const FLOODED = 3;
/** Grown over, with the flood as its nearest open pixel. */
const CANDIDATE = 4;
const JOINED = 5;

/**
 * Buffers one fill reuses across region requests: a state byte, a distance plane and a coverage byte per pixel of
 * the largest region window so far, plus per-line transform scratch.
 */
export interface RegionScratch {
  mask: Uint8Array;
  /** Flood stack for gap closing, allocated on its first use. */
  stack: Int32Array;
  distance: Float32Array;
  coverage: Uint8Array;
  line: Float64Array;
  squares: Float64Array;
  /** Per line position, the line index whose parabola gave `squares` its minimum. */
  nearest: Int32Array;
  lineKinds: Uint8Array;
  envelope: Int32Array;
  breaks: Float64Array;
}

export const createRegionScratch = (): RegionScratch => ({
  breaks: new Float64Array(0),
  coverage: new Uint8Array(0),
  distance: new Float32Array(0),
  envelope: new Int32Array(0),
  line: new Float64Array(0),
  lineKinds: new Uint8Array(0),
  mask: new Uint8Array(0),
  nearest: new Int32Array(0),
  squares: new Float64Array(0),
  stack: new Int32Array(0),
});

const ensureScratch = (scratch: RegionScratch, pixels: number, side: number) => {
  if (scratch.mask.length < pixels) {
    scratch.mask = new Uint8Array(pixels);
    scratch.distance = new Float32Array(pixels);
    scratch.coverage = new Uint8Array(pixels);
  }
  if (scratch.line.length < side) {
    scratch.line = new Float64Array(side);
    scratch.squares = new Float64Array(side);
    scratch.nearest = new Int32Array(side);
    scratch.lineKinds = new Uint8Array(side);
    scratch.envelope = new Int32Array(side);
    scratch.breaks = new Float64Array(side + 1);
  }
};

/**
 * Felzenszwalb and Huttenlocher's 1D pass: writes `min over v of (q - v)² + line[v]` for `q < length` from
 * `scratch.line` into `scratch.squares`, and the minimizing `v` into `scratch.nearest`.
 */
const lowerEnvelope = (length: number, scratch: RegionScratch) => {
  const { breaks, envelope, line, nearest, squares } = scratch;
  let k = 0;
  envelope[0] = 0;
  breaks[0] = Number.NEGATIVE_INFINITY;
  breaks[1] = Number.POSITIVE_INFINITY;
  for (let q = 1; q < length; q++) {
    const rooted = line[q] + q * q;
    let v = envelope[k];
    let s = (rooted - (line[v] + v * v)) / (2 * (q - v));
    while (s <= breaks[k]) {
      k--;
      v = envelope[k];
      s = (rooted - (line[v] + v * v)) / (2 * (q - v));
    }
    k++;
    envelope[k] = q;
    breaks[k] = s;
    breaks[k + 1] = Number.POSITIVE_INFINITY;
  }
  k = 0;
  for (let q = 0; q < length; q++) {
    while (breaks[k + 1] < q) {
      k++;
    }
    const v = envelope[k];
    squares[q] = (q - v) * (q - v) + line[v];
    nearest[q] = v;
  }
};

/**
 * Writes into `out` the squared distance from each pixel center to the nearest pixel whose `mask` byte equals
 * `feature` (0 on features; at least `DISTANCE_CAP` when no feature is that close or none exists).
 */
const squaredDistanceTransform = (
  mask: Uint8Array,
  feature: number,
  width: number,
  height: number,
  out: Float32Array,
  scratch: RegionScratch
) => {
  // Rows: 1D distance to the nearest feature in the same row.
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let last = -DISTANCE_CAP;
    for (let x = 0; x < width; x++) {
      if (mask[row + x] === feature) {
        last = x;
      }
      const gap = x - last;
      out[row + x] = gap >= 1024 ? DISTANCE_CAP : gap * gap;
    }
    last = DISTANCE_CAP;
    for (let x = width - 1; x >= 0; x--) {
      if (mask[row + x] === feature) {
        last = x;
      }
      const gap = last - x;
      if (gap < 1024 && gap * gap < out[row + x]) {
        out[row + x] = gap * gap;
      }
    }
  }
  // Columns: lower envelope of the parabolas rooted at each row distance.
  const { line, squares } = scratch;
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      line[y] = out[y * width + x];
    }
    lowerEnvelope(height, scratch);
    for (let y = 0; y < height; y++) {
      out[y * width + x] = squares[y];
    }
  }
};

/**
 * Marks INSIDE pixels OPEN when one of their four half-pixel samples lies more than `gap` half-pixels from every
 * barrier (OUTSIDE) half-pixel. On the half-pixel grid a gap `w` px wide leaves samples at most `w` half-pixels from
 * the barrier, so exactly the gaps up to `gap` px close. Row distances are the same for a pixel's two sample rows, so
 * they are stored once per pixel row (`2 × width × height` 16-bit values, the byte size of the distance plane).
 */
const markOpenPixels = (
  mask: Uint8Array,
  width: number,
  height: number,
  gap: number,
  distance: Float32Array,
  scratch: RegionScratch
) => {
  const subWidth = width * 2;
  const rows = new Uint16Array(distance.buffer, distance.byteOffset, subWidth * height);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    const subRow = y * subWidth;
    let last = -HALF_PIXEL_CAP;
    for (let s = 0; s < subWidth; s++) {
      if (mask[row + (s >> 1)] === OUTSIDE) {
        last = s;
      }
      const span = s - last;
      rows[subRow + s] = span >= 256 ? HALF_PIXEL_CAP : span * span;
    }
    last = HALF_PIXEL_CAP;
    for (let s = subWidth - 1; s >= 0; s--) {
      if (mask[row + (s >> 1)] === OUTSIDE) {
        last = s;
      }
      const span = last - s;
      if (span < 256 && span * span < rows[subRow + s]) {
        rows[subRow + s] = span * span;
      }
    }
  }
  const { line, squares } = scratch;
  const clearance = gap * gap;
  const subHeight = height * 2;
  for (let s = 0; s < subWidth; s++) {
    for (let q = 0; q < subHeight; q++) {
      line[q] = rows[(q >> 1) * subWidth + s];
    }
    lowerEnvelope(subHeight, scratch);
    const column = s >> 1;
    for (let q = 0; q < subHeight; q++) {
      const index = (q >> 1) * width + column;
      if (squares[q] > clearance && mask[index] === INSIDE) {
        mask[index] = OPEN;
      }
    }
  }
};

/**
 * Squared distance from each pixel to the nearest OPEN or FLOODED pixel into `distance`, and that pixel's state into
 * `scratch.coverage`, by the same separable transform carrying the nearest feature's state along.
 */
const nearestOpenKinds = (
  mask: Uint8Array,
  width: number,
  height: number,
  distance: Float32Array,
  scratch: RegionScratch
) => {
  const kinds = scratch.coverage;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let last = -DISTANCE_CAP;
    let kind = OUTSIDE;
    for (let x = 0; x < width; x++) {
      const state = mask[row + x];
      if (state === OPEN || state === FLOODED) {
        last = x;
        kind = state;
      }
      const gap = x - last;
      distance[row + x] = gap >= 1024 ? DISTANCE_CAP : gap * gap;
      kinds[row + x] = kind;
    }
    last = DISTANCE_CAP;
    for (let x = width - 1; x >= 0; x--) {
      const state = mask[row + x];
      if (state === OPEN || state === FLOODED) {
        last = x;
        kind = state;
      }
      const gap = last - x;
      if (gap < 1024 && gap * gap < distance[row + x]) {
        distance[row + x] = gap * gap;
        kinds[row + x] = kind;
      }
    }
  }
  const { line, lineKinds, nearest, squares } = scratch;
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      line[y] = distance[y * width + x];
      lineKinds[y] = kinds[y * width + x];
    }
    lowerEnvelope(height, scratch);
    for (let y = 0; y < height; y++) {
      distance[y * width + x] = squares[y];
      kinds[y * width + x] = lineKinds[nearest[y]];
    }
  }
};

/** Step 2 on the window's INSIDE/OUTSIDE mask; leaves the closed region as INSIDE/OUTSIDE. */
const closeRegionGaps = (
  mask: Uint8Array,
  width: number,
  height: number,
  seed: number,
  gap: number,
  distance: Float32Array,
  scratch: RegionScratch
) => {
  const pixels = width * height;
  markOpenPixels(mask, width, height, gap, distance, scratch);
  if (mask[seed] !== OPEN) {
    for (let index = 0; index < pixels; index++) {
      if (mask[index] === OPEN) {
        mask[index] = INSIDE;
      }
    }
    return;
  }
  if (scratch.stack.length < pixels) {
    scratch.stack = new Int32Array(pixels);
  }
  // Each pixel enters at most once.
  const stack = scratch.stack;
  let top = 0;
  mask[seed] = FLOODED;
  stack[top++] = seed;
  while (top > 0) {
    const index = stack[--top];
    const x = index % width;
    if (x > 0 && mask[index - 1] === OPEN) {
      mask[index - 1] = FLOODED;
      stack[top++] = index - 1;
    }
    if (x < width - 1 && mask[index + 1] === OPEN) {
      mask[index + 1] = FLOODED;
      stack[top++] = index + 1;
    }
    if (index >= width && mask[index - width] === OPEN) {
      mask[index - width] = FLOODED;
      stack[top++] = index - width;
    }
    if (index + width < pixels && mask[index + width] === OPEN) {
      mask[index + width] = FLOODED;
      stack[top++] = index + width;
    }
  }
  // Give back the margin the growth took. A grown-over pixel whose nearest open pixel is not the flood belongs to
  // the far side of a closed gap; the rest join only along steps that move strictly closer to the barrier, which
  // reach every corner but cannot climb back out of a gap into the space behind it.
  nearestOpenKinds(mask, width, height, distance, scratch);
  const kinds = scratch.coverage;
  for (let index = 0; index < pixels; index++) {
    if (mask[index] === INSIDE && kinds[index] === FLOODED) {
      mask[index] = CANDIDATE;
    }
  }
  squaredDistanceTransform(mask, OUTSIDE, width, height, distance, scratch);
  /** Joins the candidates around `index` that lie strictly closer to the barrier. */
  const joinFrom = (index: number) => {
    const x = index % width;
    const level = distance[index];
    for (let dy = -width; dy <= width; dy += width) {
      const row = index + dy;
      if (row < 0 || row >= pixels) {
        continue;
      }
      for (let dx = -1; dx <= 1; dx++) {
        const neighbor = row + dx;
        if (x + dx < 0 || x + dx >= width || mask[neighbor] !== CANDIDATE || distance[neighbor] >= level) {
          continue;
        }
        mask[neighbor] = JOINED;
        stack[top++] = neighbor;
      }
    }
  };
  for (let index = 0; index < pixels; index++) {
    if (mask[index] === FLOODED) {
      joinFrom(index);
    }
  }
  while (top > 0) {
    joinFrom(stack[--top]);
  }
  for (let index = 0; index < pixels; index++) {
    const state = mask[index];
    mask[index] = state === FLOODED || state === JOINED ? INSIDE : OUTSIDE;
  }
};

/** Coverage byte for signed boundary distance `v` (positive inside the expanded boundary). */
const coverageAt = (v: number, antiAlias: boolean): number => {
  if (!antiAlias) {
    return v > 0 ? 255 : 0;
  }
  const covered = v / 2 + 0.5;
  return covered >= 1 ? 255 : covered <= 0 ? 0 : Math.round(covered * 255);
};

const RING_EDGE_INSIDE = coverageAt(0.5, true);
const RING_EDGE_OUTSIDE = coverageAt(-0.5, true);
const RING_CORNER_INSIDE = coverageAt(Math.SQRT2 - 0.5, true);
const RING_CORNER_OUTSIDE = coverageAt(0.5 - Math.SQRT2, true);

/**
 * Anti-aliased coverage without expansion. Only distances 1 and √2 fall inside the ramp, so the 8-neighborhood
 * gives them directly and the default fill needs no distance transform. Pixels beyond the window count as the same
 * side: the window edge is either the sample edge or margin far from the region.
 */
const antiAliasRing = (mask: Uint8Array, width: number, height: number, coverage: Uint8Array) => {
  for (let y = 0; y < height; y++) {
    const row = y * width;
    const up = y > 0 ? -width : 0;
    const down = y < height - 1 ? width : 0;
    for (let x = 0; x < width; x++) {
      const index = row + x;
      const side = mask[index];
      // A missing neighbor reads the pixel itself, which never differs.
      const left = x > 0 ? -1 : 0;
      const right = x < width - 1 ? 1 : 0;
      const inside = side === INSIDE;
      if (
        mask[index + left] !== side ||
        mask[index + right] !== side ||
        mask[index + up] !== side ||
        mask[index + down] !== side
      ) {
        coverage[index] = inside ? RING_EDGE_INSIDE : RING_EDGE_OUTSIDE;
      } else if (
        mask[index + up + left] !== side ||
        mask[index + up + right] !== side ||
        mask[index + down + left] !== side ||
        mask[index + down + right] !== side
      ) {
        coverage[index] = inside ? RING_CORNER_INSIDE : RING_CORNER_OUTSIDE;
      } else {
        coverage[index] = inside ? 255 : 0;
      }
    }
  }
};

/** Quantized threshold for a 0..100 tolerance. */
export const fillThreshold = (tolerance: number): number => {
  const scaled = Math.round((tolerance * (FILL_LEVELS - 1)) / 100);
  return scaled < 0 ? 0 : scaled > FILL_LEVELS - 1 ? FILL_LEVELS - 1 : scaled;
};

/** Computes the coverage for `params` from a level map; `scratch` carries buffers between calls for one fill. */
export const computeFillRegion = (
  map: FillLevelMap,
  params: FillRegionParams,
  scratch: RegionScratch = createRegionScratch()
): FillRegion => {
  const { height, levelBounds, levels, width } = map;
  const threshold = fillThreshold(params.tolerance);
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let level = 0; level <= threshold; level++) {
    const at = level * 4;
    if (levelBounds[at] > levelBounds[at + 2]) {
      continue;
    }
    minX = Math.min(minX, levelBounds[at]);
    minY = Math.min(minY, levelBounds[at + 1]);
    maxX = Math.max(maxX, levelBounds[at + 2]);
    maxY = Math.max(maxY, levelBounds[at + 3]);
  }
  if (maxX < minX) {
    return EMPTY_FILL_REGION;
  }

  const { antiAlias, expand } = params;
  // Expansion and the anti-alias ramp reach `expand + 1.5` px past the region; one more pixel keeps a ring of
  // outside pixels in the window, so distances to "outside" never depend on pixels beyond it.
  const margin = Math.ceil(Math.max(expand, 0)) + 2;
  const left = Math.max(0, minX - margin);
  const top = Math.max(0, minY - margin);
  const windowWidth = Math.min(width, maxX + margin + 1) - left;
  const windowHeight = Math.min(height, maxY + margin + 1) - top;
  const pixels = windowWidth * windowHeight;
  ensureScratch(scratch, pixels, Math.max(windowWidth, windowHeight) * 2);
  const mask = scratch.mask.subarray(0, pixels);
  const coverage = scratch.coverage.subarray(0, pixels);
  const distance = scratch.distance.subarray(0, pixels);

  for (let y = 0; y < windowHeight; y++) {
    const source = (top + y) * width + left;
    const target = y * windowWidth;
    for (let x = 0; x < windowWidth; x++) {
      mask[target + x] = levels[source + x] <= threshold ? INSIDE : OUTSIDE;
    }
  }

  if (map.contiguous && params.closeGaps > 0) {
    const seed = (map.seedY - top) * windowWidth + (map.seedX - left);
    closeRegionGaps(mask, windowWidth, windowHeight, seed, params.closeGaps, distance, scratch);
  }

  if (expand === 0 && !antiAlias) {
    for (let index = 0; index < pixels; index++) {
      coverage[index] = mask[index] === INSIDE ? 255 : 0;
    }
  } else if (expand === 0) {
    antiAliasRing(mask, windowWidth, windowHeight, coverage);
  } else {
    // v = signed distance from the expanded boundary: inside `D_in - 0.5 + e`, outside `0.5 - D_out + e`, where
    // D is the center distance to the nearest pixel of the other side.
    coverage.fill(0);
    if (expand > 0) {
      squaredDistanceTransform(mask, INSIDE, windowWidth, windowHeight, distance, scratch);
      for (let index = 0; index < pixels; index++) {
        coverage[index] =
          mask[index] === INSIDE ? 255 : coverageAt(0.5 - Math.sqrt(distance[index]) + expand, antiAlias);
      }
    } else {
      squaredDistanceTransform(mask, OUTSIDE, windowWidth, windowHeight, distance, scratch);
      for (let index = 0; index < pixels; index++) {
        if (mask[index] === INSIDE) {
          coverage[index] = coverageAt(Math.sqrt(distance[index]) - 0.5 + expand, antiAlias);
        }
      }
    }
  }

  let coveredMinX = windowWidth;
  let coveredMinY = windowHeight;
  let coveredMaxX = -1;
  let coveredMaxY = -1;
  let pixelCount = 0;
  for (let y = 0; y < windowHeight; y++) {
    const row = y * windowWidth;
    for (let x = 0; x < windowWidth; x++) {
      if (coverage[row + x] !== 0) {
        pixelCount++;
        if (x < coveredMinX) {
          coveredMinX = x;
        }
        if (x > coveredMaxX) {
          coveredMaxX = x;
        }
        if (y < coveredMinY) {
          coveredMinY = y;
        }
        coveredMaxY = y;
      }
    }
  }
  if (pixelCount === 0) {
    return EMPTY_FILL_REGION;
  }
  const boundsWidth = coveredMaxX - coveredMinX + 1;
  const boundsHeight = coveredMaxY - coveredMinY + 1;
  const result = new Uint8Array(boundsWidth * boundsHeight);
  for (let y = 0; y < boundsHeight; y++) {
    const start = (coveredMinY + y) * windowWidth + coveredMinX;
    result.set(coverage.subarray(start, start + boundsWidth), y * boundsWidth);
  }
  return {
    bounds: { height: boundsHeight, width: boundsWidth, x: left + coveredMinX, y: top + coveredMinY },
    coverage: result,
    pixelCount,
  };
};

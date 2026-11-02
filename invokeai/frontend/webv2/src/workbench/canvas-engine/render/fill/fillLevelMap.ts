// SPDX-License-Identifier: Apache-2.0
/**
 * The fill level map: for every sample pixel, the smallest quantized tolerance (0..255) at which a fill from the seed
 * takes it, so any tolerance is a threshold `level ≤ t` on one retained byte per pixel.
 *
 * Invariants:
 * - `levels[seed] = 0`, and a pixel identical to the seed color has level 0 in non-contiguous mode.
 * - Contiguous: `levels[p]` is the bottleneck cost of the cheapest 4-connected path from the seed to `p`, the path's
 *   largest seed-color distance. So `{ level ≤ t }` is the 4-connected component of `{ distance ≤ t }` holding the
 *   seed, it nests in `t`, and `t = 255` takes the whole sample.
 * - Non-contiguous: `levels[p]` is `p`'s own seed-color distance.
 * - `levelCounts[l]` counts the pixels at level `l`; `levelBounds` holds their inclusive bounds, so a threshold's
 *   pixel count and bounds come from 256 entries instead of a pass over the map.
 */

import { createSeedColorDistance } from '@workbench/canvas-engine/render/fill/colorDistance';

/** Straight (unpremultiplied) RGBA8 pixels, as `getImageData` returns them. */
export interface FillSample {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;
}

export interface FillLevelMap {
  readonly width: number;
  readonly height: number;
  readonly seedX: number;
  readonly seedY: number;
  readonly contiguous: boolean;
  readonly levels: Uint8Array;
  readonly levelCounts: Uint32Array;
  /** Per level, inclusive `minX, minY, maxX, maxY`; an empty level keeps `minX > maxX`. */
  readonly levelBounds: Int32Array;
}

export const FILL_LEVELS = 256;
const UNVISITED = -2;
const END = -1;

/** Per-level pixel counts and bounds in one row-major pass. */
const summarizeLevels = (levels: Uint8Array, width: number, height: number) => {
  const levelCounts = new Uint32Array(FILL_LEVELS);
  const levelBounds = new Int32Array(FILL_LEVELS * 4);
  for (let level = 0; level < FILL_LEVELS; level++) {
    levelBounds[level * 4] = width;
    levelBounds[level * 4 + 1] = height;
    levelBounds[level * 4 + 2] = -1;
    levelBounds[level * 4 + 3] = -1;
  }
  for (let y = 0, index = 0; y < height; y++) {
    for (let x = 0; x < width; x++, index++) {
      const at = levels[index] * 4;
      levelCounts[levels[index]]++;
      if (levelBounds[at + 1] > y) {
        levelBounds[at + 1] = y;
      }
      levelBounds[at + 3] = y;
      if (x < levelBounds[at]) {
        levelBounds[at] = x;
      }
      if (x > levelBounds[at + 2]) {
        levelBounds[at + 2] = x;
      }
    }
  }
  return { levelBounds, levelCounts };
};

/** Builds the level map for a fill from `(seedX, seedY)`; throws a `RangeError` for a malformed sample or seed. */
export const buildFillLevelMap = (
  sample: FillSample,
  seedX: number,
  seedY: number,
  contiguous: boolean
): FillLevelMap => {
  const { data, height, width } = sample;
  const pixelCount = width * height;
  if (!Number.isInteger(width) || !Number.isInteger(height) || pixelCount <= 0 || data.length !== pixelCount * 4) {
    throw new RangeError('Fill sample must hold width * height RGBA8 pixels.');
  }
  if (
    !Number.isInteger(seedX) ||
    !Number.isInteger(seedY) ||
    seedX < 0 ||
    seedY < 0 ||
    seedX >= width ||
    seedY >= height
  ) {
    throw new RangeError('Fill seed must be an integer pixel inside the sample.');
  }

  const distance = createSeedColorDistance(data, (seedY * width + seedX) * 4);
  const levels = new Uint8Array(pixelCount);
  if (!contiguous) {
    for (let index = 0; index < pixelCount; index++) {
      levels[index] = distance(data, index * 4);
    }
    return { contiguous, height, levels, seedX, seedY, width, ...summarizeLevels(levels, width, height) };
  }

  // Priority flood over 256 FIFO buckets chained through `next`. Buckets pop in increasing level, so the first push
  // of a pixel already carries its bottleneck level and every pixel is pushed and measured exactly once.
  const next = new Int32Array(pixelCount).fill(UNVISITED);
  const tails = new Int32Array(FILL_LEVELS).fill(END);
  const heads = new Int32Array(FILL_LEVELS).fill(END);
  let current = 0;
  const visit = (index: number) => {
    if (next[index] !== UNVISITED) {
      return;
    }
    const own = distance(data, index * 4);
    const level = own > current ? own : current;
    levels[index] = level;
    next[index] = END;
    const tail = tails[level];
    if (tail === END) {
      heads[level] = index;
    } else {
      next[tail] = index;
    }
    tails[level] = index;
  };

  visit(seedY * width + seedX);
  for (; current < FILL_LEVELS; current++) {
    for (let index = heads[current]; index !== END; index = next[index]) {
      const x = index % width;
      if (x > 0) {
        visit(index - 1);
      }
      if (x < width - 1) {
        visit(index + 1);
      }
      if (index >= width) {
        visit(index - width);
      }
      if (index + width < pixelCount) {
        visit(index + width);
      }
    }
  }
  return { contiguous, height, levels, seedX, seedY, width, ...summarizeLevels(levels, width, height) };
};

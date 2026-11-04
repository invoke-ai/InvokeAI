// SPDX-License-Identifier: Apache-2.0
/** Small deterministic fill samples for the fill kernel, service and worker tests. */

import type { FillSample } from '@workbench/canvas-engine/render/fill/fillLevelMap';

export type Rgba = readonly [number, number, number, number];

export const WHITE: Rgba = [255, 255, 255, 255];
export const INK: Rgba = [20, 20, 30, 255];
export const CLEAR: Rgba = [0, 0, 0, 0];

export const createSample = (width: number, height: number, color: Rgba): FillSample => {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < data.length; index += 4) {
    data.set(color, index);
  }
  return { data, height, width };
};

export const paintRect = (
  sample: FillSample,
  x: number,
  y: number,
  width: number,
  height: number,
  color: Rgba
): void => {
  for (let row = y; row < y + height; row++) {
    for (let column = x; column < x + width; column++) {
      sample.data.set(color, (row * sample.width + column) * 4);
    }
  }
};

/** Pixel indices whose predicate holds, for set comparisons. */
export const indicesWhere = (count: number, predicate: (index: number) => boolean): number[] => {
  const result: number[] = [];
  for (let index = 0; index < count; index++) {
    if (predicate(index)) {
      result.push(index);
    }
  }
  return result;
};

/** Deterministic mulberry32. */
export const seeded = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

/** A smooth two-axis color gradient with mild noise, like a photo's sky. */
export const createGradientSample = (width: number, height: number): FillSample => {
  const random = seeded(0xf111);
  const sample = createSample(width, height, WHITE);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const noise = random() * 6 - 3;
      sample.data.set(
        [(x / width) * 180 + 40 + noise, (y / height) * 120 + 80 + noise, 200 - (x / width) * 60 + noise, 255],
        (y * width + x) * 4
      );
    }
  }
  return sample;
};

/**
 * White line art, 46 × 26: two 2 px ink box outlines, left one with a 2 px gap in its top edge, right one with a
 * 6 px gap. Each interior spans 12 × 12 px from (`x` + 2, 6).
 */
export const LINE_ART = {
  height: 26,
  narrow: { gap: { width: 2, x: 9 }, interior: { height: 12, width: 12, x: 4, y: 6 }, seed: { x: 10, y: 12 } },
  outside: { x: 0, y: 0 },
  wide: { gap: { width: 6, x: 29 }, interior: { height: 12, width: 12, x: 26, y: 6 }, seed: { x: 32, y: 12 } },
  width: 46,
} as const;

export const createLineArtSample = (): FillSample => {
  const sample = createSample(LINE_ART.width, LINE_ART.height, WHITE);
  for (const box of [LINE_ART.narrow, LINE_ART.wide]) {
    const left = box.interior.x - 2;
    const top = box.interior.y - 2;
    const size = box.interior.width + 4;
    paintRect(sample, left, top, size, 2, INK);
    paintRect(sample, left, top + size - 2, size, 2, INK);
    paintRect(sample, left, top, 2, size, INK);
    paintRect(sample, left + size - 2, top, 2, size, INK);
    paintRect(sample, box.gap.x, top, box.gap.width, 2, WHITE);
  }
  return sample;
};

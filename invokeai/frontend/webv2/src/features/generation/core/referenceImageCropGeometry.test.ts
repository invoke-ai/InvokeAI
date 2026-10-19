import { describe, expect, it } from 'vitest';

import {
  constrainReferenceCropBox,
  matchReferenceCropAspectRatio,
  REFERENCE_CROP_ASPECT_RATIOS,
  resolveReferenceCropDrag,
  type ReferenceCropConstraints,
  type ReferenceCropDragMode,
} from './referenceImage';

const IMAGE = { height: 600, width: 800 };
const constraints = (ratio: number | null, image = IMAGE): ReferenceCropConstraints => ({
  image,
  minDimensionPx: 64,
  ratio,
});
/** The pixel ratio a percent box has on an image. */
const pixelRatio = (box: { height: number; width: number }, image = IMAGE): number =>
  (box.width * image.width) / (box.height * image.height);
const drag = (
  mode: ReferenceCropDragMode,
  box: { height: number; width: number; x: number; y: number },
  from: { x: number; y: number },
  to: { x: number; y: number },
  ratio: number | null
) => resolveReferenceCropDrag({ box, mode, ...from }, to, constraints(ratio));
const expectInside = (box: { height: number; width: number; x: number; y: number }) => {
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.y).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(100 + 1e-9);
  expect(box.y + box.height).toBeLessThanOrEqual(100 + 1e-9);
};

describe('constrainReferenceCropBox', () => {
  it('reshapes a box to the preset ratio around its center while keeping its area', () => {
    const box = { height: 50, width: 25, x: 50, y: 25 };
    const square = constrainReferenceCropBox(box, constraints(1));

    expect(pixelRatio(square)).toBeCloseTo(1);
    expect(square.width * square.height).toBeCloseTo(box.width * box.height);
    expect(square.x + square.width / 2).toBeCloseTo(62.5);
    expect(square.y + square.height / 2).toBeCloseTo(50);
  });

  it('shrinks and slides a box that would otherwise leave the image, and leaves a free box alone', () => {
    const box = { height: 100, width: 100, x: 0, y: 0 };
    const wide = constrainReferenceCropBox(box, constraints(16 / 9));

    expect(pixelRatio(wide)).toBeCloseTo(16 / 9);
    expect(wide.width).toBe(100);
    expectInside(wide);
    expect(constrainReferenceCropBox(box, constraints(null))).toBe(box);
  });

  it('never drops an edge below the minimum pixel size', () => {
    const tiny = constrainReferenceCropBox({ height: 2, width: 2, x: 10, y: 10 }, constraints(16 / 9));

    expect(tiny.height * IMAGE.height).toBeGreaterThanOrEqual(64 * 100 - 1e-6);
    expect(pixelRatio(tiny)).toBeCloseTo(16 / 9);
  });
});

describe('resolveReferenceCropDrag', () => {
  const box = { height: 40, width: 40, x: 30, y: 30 };

  it('moves within the image and resizes freely from any handle', () => {
    expect(drag('move', box, { x: 50, y: 50 }, { x: 90, y: 20 }, null)).toEqual({ ...box, x: 60, y: 0 });
    expect(drag('se', box, { x: 70, y: 70 }, { x: 80, y: 60 }, null)).toEqual({ height: 30, width: 50, x: 30, y: 30 });
    expect(drag('nw', box, { x: 30, y: 30 }, { x: 0, y: 0 }, null)).toEqual({ height: 70, width: 70, x: 0, y: 0 });
  });

  it('keeps a corner drag on the preset ratio, anchored at the opposite corner', () => {
    // Pixel-square on a 4:3 image: 50% of the width is 400px, so the height becomes 400px, two thirds of 600.
    const grown = drag('se', box, { x: 70, y: 70 }, { x: 80, y: 75 }, 1);

    expect(pixelRatio(grown)).toBeCloseTo(1);
    expect(grown.x).toBe(30);
    expect(grown.y).toBe(30);
    expect(grown.width).toBeCloseTo(50);
    expect(grown.height).toBeCloseTo(66.667, 2);
    expectInside(grown);
  });

  it('grows the other dimension around the center line for an edge drag', () => {
    const wider = drag('e', box, { x: 70, y: 50 }, { x: 80, y: 50 }, 1);

    expect(pixelRatio(wider)).toBeCloseTo(1);
    expect(wider.x).toBe(30);
    expect(wider.width).toBeCloseTo(50);
    expect(wider.y + wider.height / 2).toBeCloseTo(50);
  });

  it.each([
    ['e', { height: 40, width: 40, x: 0, y: 0 }, { x: 40, y: 20 }, { x: 100, y: 20 }],
    ['w', { height: 40, width: 40, x: 60, y: 60 }, { x: 60, y: 80 }, { x: 0, y: 80 }],
    ['s', { height: 40, width: 40, x: 0, y: 0 }, { x: 20, y: 40 }, { x: 20, y: 100 }],
    ['n', { height: 40, width: 40, x: 60, y: 60 }, { x: 80, y: 60 }, { x: 80, y: 0 }],
  ] as const)('lets the %s side grow to the full image by sliding the other axis', (mode, box, from, to) => {
    const grown = resolveReferenceCropDrag({ box, mode, ...from }, to, constraints(1, { width: 800, height: 800 }));
    expect(grown).toEqual({ x: 0, y: 0, width: 100, height: 100 });
  });

  it.each(['nw', 'ne', 'sw', 'se'] as const)('shrinks the %s corner with either single-axis gesture', (mode) => {
    const image = { width: 600, height: 800 };
    // A 240px square on a portrait source.
    const box = { x: 30, y: 30, width: 40, height: 30 };
    const from = { x: mode.includes('w') ? 30 : 70, y: mode.includes('n') ? 30 : 60 };
    for (const to of [
      { ...from, x: from.x + (mode.includes('w') ? 10 : -10) },
      { ...from, y: from.y + (mode.includes('n') ? 7.5 : -7.5) },
    ]) {
      const shrunk = resolveReferenceCropDrag({ box, mode, ...from }, to, constraints(1, image));
      expect(shrunk.width).toBeCloseTo(30);
      expect(shrunk.height).toBeCloseTo(22.5);
      expect(shrunk.x + (mode.includes('w') ? shrunk.width : 0)).toBeCloseTo(mode.includes('w') ? 70 : 30);
      expect(shrunk.y + (mode.includes('n') ? shrunk.height : 0)).toBeCloseTo(mode.includes('n') ? 60 : 30);
      expectInside(shrunk);
    }
  });

  it('stops a ratio-locked drag at the image edge instead of breaking the ratio', () => {
    const pinned = drag('se', box, { x: 70, y: 70 }, { x: 100, y: 100 }, 16 / 9);

    expect(pixelRatio(pinned)).toBeCloseTo(16 / 9);
    expectInside(pinned);
    expect(pinned.x + pinned.width).toBeCloseTo(100);
  });
});

describe('matchReferenceCropAspectRatio', () => {
  it('recognizes every preset from a pixel-rounded box and reads anything else as free', () => {
    for (const { id, ratio } of REFERENCE_CROP_ASPECT_RATIOS) {
      if (ratio === null) {
        continue;
      }
      const heightPx = 300;
      const widthPx = Math.round(heightPx * ratio);
      const box = { height: (heightPx / 600) * 100, width: (widthPx / 800) * 100, x: 0, y: 0 };
      expect(matchReferenceCropAspectRatio(box, IMAGE)).toBe(id);
    }
    expect(matchReferenceCropAspectRatio({ height: 50, width: 37, x: 0, y: 0 }, IMAGE)).toBe('free');
  });
});

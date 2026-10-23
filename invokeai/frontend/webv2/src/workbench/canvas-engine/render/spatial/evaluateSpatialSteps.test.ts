import type { SpatialAdjustmentStep } from '@workbench/canvas-engine/render/spatial/spatialSteps';

import { evaluateSpatialSteps } from '@workbench/canvas-engine/render/spatial/evaluateSpatialSteps';
import {
  MAX_SPATIAL_PIXELS,
  SPATIAL_SCRATCH_BYTES_PER_PIXEL,
} from '@workbench/canvas-engine/render/spatial/spatialSteps';
import { describe, expect, it, vi } from 'vitest';

import parity from './__fixtures__/spatial_adjustment_parity.json';

const decode = (text: string): Uint8ClampedArray => Uint8ClampedArray.from(atob(text), (char) => char.charCodeAt(0));

const cases = parity.adjustments.map((entry) => ({
  ...entry,
  steps: entry.steps as SpatialAdjustmentStep[],
}));

/** Float32 rounding differs between torch and these kernels, so visible color may land one byte away. */
const compareWithPython = (actual: Uint8ClampedArray, input: Uint8ClampedArray, expected: Uint8ClampedArray) => {
  let alphaChanged = 0;
  let hiddenChanged = 0;
  let visibleChannels = 0;
  let differing = 0;
  let maxDifference = 0;
  for (let p = 0; p < actual.length; p += 4) {
    if (actual[p + 3] !== input[p + 3]) {
      alphaChanged++;
    }
    for (let c = 0; c < 3; c++) {
      if (input[p + 3] === 0) {
        hiddenChanged += actual[p + c] === input[p + c] ? 0 : 1;
        continue;
      }
      visibleChannels++;
      const difference = Math.abs(actual[p + c] - expected[p + c]);
      differing += difference === 0 ? 0 : 1;
      maxDifference = Math.max(maxDifference, difference);
    }
  }
  return { alphaChanged, differingShare: differing / visibleChannels, hiddenChanged, maxDifference };
};

const evaluate = (pixels: Uint8ClampedArray, width: number, height: number, steps: SpatialAdjustmentStep[]) => {
  const output = pixels.slice();
  expect(evaluateSpatialSteps(output, width, height, steps)).toEqual({ status: 'ok' });
  return output;
};

describe('evaluateSpatialSteps', () => {
  it.each(cases)('matches the Python node output for $name', ({ expected, height, rgba, steps, width }) => {
    const input = decode(rgba);
    const result = compareWithPython(evaluate(input, width, height, steps), input, decode(expected));

    expect(result.alphaChanged).toBe(0);
    expect(result.hiddenChanged).toBe(0);
    expect(result.maxDifference).toBeLessThanOrEqual(1);
    expect(result.differingShare).toBeLessThanOrEqual(0.01);
  });

  it('gives content identical results whatever transparent padding surrounds it', () => {
    const { height, rgba, steps, width } = cases.find((entry) => entry.name === 'clarity_then_dehaze')!;
    const content = decode(rgba);
    const padded = new Uint8ClampedArray((width + 11) * (height + 6) * 4);
    for (let i = 0; i < padded.length; i += 4) {
      // Hidden padding colors are arbitrary and must survive untouched.
      padded[i] = i % 251;
      padded[i + 1] = 7;
      padded[i + 2] = 200;
    }
    const left = 7;
    const top = 4;
    const paddedWidth = width + 11;
    for (let y = 0; y < height; y++) {
      padded.set(content.subarray(y * width * 4, (y + 1) * width * 4), ((top + y) * paddedWidth + left) * 4);
    }

    const alone = evaluate(content, width, height, steps);
    const surrounded = evaluate(padded, paddedWidth, height + 6, steps);

    const expected = padded.slice();
    for (let y = 0; y < height; y++) {
      expected.set(alone.subarray(y * width * 4, (y + 1) * width * 4), ((top + y) * paddedWidth + left) * 4);
    }
    expect(surrounded).toEqual(expected);
  });

  it('clamps out-of-range amounts and sizes to the node limits', () => {
    const { height, rgba, width } = cases[0];
    const input = decode(rgba);

    expect(evaluate(input, width, height, [{ amount: 500, kind: 'clarity', size: 4000 }])).toEqual(
      evaluate(input, width, height, [{ amount: 200, kind: 'clarity', size: 1000 }])
    );
    expect(evaluate(input, width, height, [{ amount: -900, kind: 'texture', size: 0.2 }])).toEqual(
      evaluate(input, width, height, [{ amount: -200, kind: 'texture', size: 1 }])
    );
    expect(evaluate(input, width, height, [{ amount: 1e9, kind: 'dehaze' }])).toEqual(
      evaluate(input, width, height, [{ amount: 200, kind: 'dehaze' }])
    );
  });

  it('leaves pixels untouched for zero or non-finite steps and for fully transparent layers', () => {
    const { height, rgba, width } = cases[0];
    const input = decode(rgba);
    const identitySteps: SpatialAdjustmentStep[] = [
      { amount: 0, kind: 'clarity', size: 40 },
      { amount: Number.NaN, kind: 'texture', size: 8 },
      { amount: 50, kind: 'texture', size: Number.POSITIVE_INFINITY },
      { amount: Number.NEGATIVE_INFINITY, kind: 'dehaze' },
    ];
    expect(evaluate(input, width, height, identitySteps)).toEqual(input);

    const transparent = input.map((value, index) => (index % 4 === 3 ? 0 : value));
    expect(evaluate(transparent, width, height, [{ amount: 100, kind: 'dehaze' }])).toEqual(transparent);
  });

  it('allocates the per-pixel scratch the service reserves, less the labels when nothing is hidden', () => {
    const allocations: number[] = [];
    vi.stubGlobal(
      'ArrayBuffer',
      class extends ArrayBuffer {
        constructor(length: number) {
          super(length);
          allocations.push(length);
        }
      }
    );
    const withHidden = cases.find((entry) => entry.name === 'clarity_default_size')!;
    const opaque = cases.find((entry) => entry.name === 'flat_zero_channel_clarity')!;
    try {
      evaluate(decode(withHidden.rgba), withHidden.width, withHidden.height, withHidden.steps);
      evaluate(decode(opaque.rgba), opaque.width, opaque.height, opaque.steps);
    } finally {
      vi.unstubAllGlobals();
    }

    expect(allocations).toEqual([
      withHidden.width * withHidden.height * SPATIAL_SCRATCH_BYTES_PER_PIXEL,
      opaque.width * opaque.height * (SPATIAL_SCRATCH_BYTES_PER_PIXEL - 4),
    ]);
  });

  it('refuses alpha bounds larger than the pixel limit without touching pixels', () => {
    const width = 4097;
    const height = MAX_SPATIAL_PIXELS / 4096;
    const pixels = new Uint8ClampedArray(width * height * 4);
    const last = pixels.length - 4;
    pixels.set([10, 20, 30, 255], 0);
    pixels.set([40, 50, 60, 255], last);

    expect(evaluateSpatialSteps(pixels, width, height, [{ amount: 100, kind: 'dehaze' }])).toEqual({
      status: 'too-large',
    });
    expect(Array.from(pixels.subarray(0, 4))).toEqual([10, 20, 30, 255]);
    expect(Array.from(pixels.subarray(last))).toEqual([40, 50, 60, 255]);
  });
});

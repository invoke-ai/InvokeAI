import { boxMean, edgeAwareBand } from '@workbench/canvas-engine/render/spatial/guidedFilter';
import { describe, expect, it } from 'vitest';

import fixtures from './__fixtures__/shared_image_manipulation_numerics.json';

const plane = (length: number) => new Float32Array(length);

/** OpenCV's BORDER_REFLECT_101 by repeated mirroring, independent of the kernels' periodic form. */
const mirror = (index: number, length: number): number => {
  if (length === 1) {
    return 0;
  }
  let position = index;
  while (position < 0 || position >= length) {
    position = position < 0 ? -position : 2 * (length - 1) - position;
  }
  return position;
};

const directBoxMean = (values: Float32Array, width: number, height: number, radius: number, squared: boolean) =>
  Array.from({ length: width * height }, (_, index) => {
    const x = index % width;
    const y = Math.floor(index / width);
    let sum = 0;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const value = values[mirror(y + dy, height) * width + mirror(x + dx, width)];
        sum += squared ? value * value : value;
      }
    }
    return sum / (2 * radius + 1) ** 2;
  });

describe('boxMean', () => {
  it.each([
    { height: 5, radius: 2, width: 6 },
    { height: 5, radius: 7, width: 4 },
    { height: 1, radius: 9, width: 6 },
    { height: 3, radius: 4, width: 1 },
  ])('matches reflected direct sums for $width x $height at radius $radius', ({ height, radius, width }) => {
    const source = Float32Array.from({ length: width * height }, (_, index) => Math.sin(index * 1.7) * 0.5 + 0.5);
    const columnSums = new Float64Array(width);
    for (const squared of [false, true]) {
      const target = plane(width * height);
      boxMean(source, target, width, height, radius, squared, columnSums);
      directBoxMean(source, width, height, radius, squared).forEach((expected, index) => {
        expect(target[index]).toBeCloseTo(expected, 6);
      });
    }
  });
});

describe('edgeAwareBand', () => {
  it('matches the independently derived band whose fine scale is the input itself', () => {
    const { atol, epsilon, expected, inner_radius, lightness, outer_radius } = fixtures.multiscale_band;
    const width = lightness[0].length;
    const height = lightness.length;
    const count = width * height;
    const detail = plane(count);
    const scratch = {
      columnSums: new Float64Array(width),
      first: plane(count),
      second: plane(count),
      third: plane(count),
    };

    edgeAwareBand(
      Float32Array.from(lightness.flat()),
      detail,
      width,
      height,
      inner_radius,
      outer_radius,
      epsilon,
      scratch
    );

    expected.flat().forEach((value, index) => expect(Math.abs(detail[index] - value)).toBeLessThanOrEqual(atol));
  });
});

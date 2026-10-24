// SPDX-License-Identifier: Apache-2.0
export type ScalarImage = {
  readonly width: number;
  readonly height: number;
  readonly data: Float64Array;
};

const validateImage = (image: ScalarImage, name: string): void => {
  if (
    !Number.isInteger(image.width) ||
    image.width < 1 ||
    !Number.isInteger(image.height) ||
    image.height < 1 ||
    !(image.data instanceof Float64Array) ||
    image.data.length !== image.width * image.height
  ) {
    throw new TypeError(`${name} must be a non-empty image with width * height Float64 samples`);
  }
  for (let i = 0; i < image.data.length; i++) {
    if (!Number.isFinite(image.data[i])) {
      throw new TypeError(`${name} must contain finite samples`);
    }
  }
};

const reflect101 = (index: number, length: number): number => {
  if (length <= 1) {
    return 0;
  }
  const period = 2 * length - 2;
  const wrapped = ((index % period) + period) % period;
  return wrapped < length ? wrapped : period - wrapped;
};

/** Computes a normalized square box mean with OpenCV BORDER_REFLECT_101 boundaries. */
const boxMean = (
  first: Float64Array,
  width: number,
  height: number,
  radius: number,
  second?: Float64Array,
  square = false
): Float64Array => {
  const valueAt = (index: number) =>
    square ? first[index] * first[index] : second ? first[index] * second[index] : first[index];
  const windowSize = radius * 2 + 1;
  const horizontal = new Float64Array(first.length);
  const output = new Float64Array(first.length);
  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    let sum = 0;
    for (let offset = -radius; offset <= radius; offset++) {
      const index = rowOffset + reflect101(offset, width);
      sum += valueAt(index);
    }
    for (let x = 0; x < width; x++) {
      horizontal[rowOffset + x] = sum / windowSize;
      const leaving = rowOffset + reflect101(x - radius, width);
      const entering = rowOffset + reflect101(x + radius + 1, width);
      sum -= valueAt(leaving);
      sum += valueAt(entering);
    }
  }
  for (let x = 0; x < width; x++) {
    let sum = 0;
    for (let offset = -radius; offset <= radius; offset++) {
      const index = reflect101(offset, height) * width + x;
      sum += horizontal[index];
    }
    for (let y = 0; y < height; y++) {
      const index = y * width + x;
      output[index] = sum / windowSize;
      sum -= horizontal[reflect101(y - radius, height) * width + x];
      sum += horizontal[reflect101(y + radius + 1, height) * width + x];
    }
  }
  return output;
};

/** Applies the classic single-channel guided filter to equally sized row-major images. */
export const guidedFilter = (guide: ScalarImage, signal: ScalarImage, radius: number, epsilon: number): ScalarImage => {
  validateImage(guide, 'guide');
  validateImage(signal, 'signal');
  if (guide.width !== signal.width || guide.height !== signal.height) {
    throw new Error('guide and signal must have the same dimensions');
  }
  if (!Number.isInteger(radius) || radius < 0) {
    throw new Error('radius must be a non-negative integer');
  }
  if (!Number.isFinite(epsilon) || epsilon <= 0) {
    throw new Error('epsilon must be a finite positive number');
  }
  const { width, height } = guide;
  const count = guide.data.length;
  if (radius === 0) {
    return { width, height, data: signal.data.slice() };
  }

  const meanGuide = boxMean(guide.data, width, height, radius);
  const meanSignal = boxMean(signal.data, width, height, radius);
  const meanGuideSquared = boxMean(guide.data, width, height, radius, undefined, true);
  const meanGuideSignal = boxMean(guide.data, width, height, radius, signal.data);
  const a = new Float64Array(count);
  const b = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const variance = Math.max(0, meanGuideSquared[i] - meanGuide[i] * meanGuide[i]);
    const covariance = meanGuideSignal[i] - meanGuide[i] * meanSignal[i];
    a[i] = covariance / (variance + epsilon);
    b[i] = meanSignal[i] - a[i] * meanGuide[i];
  }
  const meanA = boxMean(a, width, height, radius);
  const meanB = boxMean(b, width, height, radius);
  const data = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    data[i] = meanA[i] * guide.data[i] + meanB[i];
  }
  return { width, height, data };
};

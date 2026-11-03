// SPDX-License-Identifier: Apache-2.0
/**
 * Self-guided filtering and edge-aware bands with the backend's numerics (`guided_filter.py`, `multiscale.py`):
 * normalized box means with OpenCV `BORDER_REFLECT_101` borders at any radius, accumulated in float64 as OpenCV's
 * float32 box filter does. Work is O(width × height) for every radius.
 */

/** Scratch shared by the filters: three planes of the image's size and one float64 row. */
export type FilterScratch = {
  readonly first: Float32Array;
  readonly second: Float32Array;
  readonly third: Float32Array;
  readonly columnSums: Float64Array;
};

const reflect101 = (index: number, length: number): number => {
  if (length === 1) {
    return 0;
  }
  const period = 2 * length - 2;
  const wrapped = ((index % period) + period) % period;
  return wrapped < length ? wrapped : period - wrapped;
};

/**
 * Reflect-101 bookkeeping along one axis: the distinct indices of the first window (centered on 0) with their
 * multiplicities, and the indices entering and leaving as the window moves from each position to the next.
 */
type AxisWindow = {
  readonly firstIndices: Int32Array;
  readonly firstCounts: Int32Array;
  readonly entering: Int32Array;
  readonly leaving: Int32Array;
};

const axisWindow = (length: number, radius: number): AxisWindow => {
  const counts = new Int32Array(length);
  for (let offset = -radius; offset <= radius; offset++) {
    counts[reflect101(offset, length)]++;
  }
  let distinct = 0;
  for (let index = 0; index < length; index++) {
    if (counts[index] !== 0) {
      distinct++;
    }
  }
  const firstIndices = new Int32Array(distinct);
  const firstCounts = new Int32Array(distinct);
  for (let index = 0, slot = 0; index < length; index++) {
    if (counts[index] !== 0) {
      firstIndices[slot] = index;
      firstCounts[slot] = counts[index];
      slot++;
    }
  }
  const entering = new Int32Array(length);
  const leaving = new Int32Array(length);
  for (let position = 0; position < length; position++) {
    entering[position] = reflect101(position + radius + 1, length);
    leaving[position] = reflect101(position - radius, length);
  }
  return { entering, firstCounts, firstIndices, leaving };
};

/**
 * Writes the normalized (2r+1)² box mean of `source`, or of its squares, to `target` (a different plane). Columns are
 * summed while streaming rows, then each row slides a window across the column sums.
 */
export const boxMean = (
  source: Float32Array,
  target: Float32Array,
  width: number,
  height: number,
  radius: number,
  squared: boolean,
  columnSums: Float64Array
): void => {
  const rows = axisWindow(height, radius);
  const columns = axisWindow(width, radius);
  const scale = 1 / ((2 * radius + 1) * (2 * radius + 1));

  columnSums.fill(0, 0, width);
  for (let slot = 0; slot < rows.firstIndices.length; slot++) {
    const offset = rows.firstIndices[slot] * width;
    const count = rows.firstCounts[slot];
    for (let x = 0; x < width; x++) {
      const value = source[offset + x];
      columnSums[x] += count * (squared ? value * value : value);
    }
  }

  for (let y = 0; y < height; y++) {
    let sum = 0;
    for (let slot = 0; slot < columns.firstIndices.length; slot++) {
      sum += columns.firstCounts[slot] * columnSums[columns.firstIndices[slot]];
    }
    const rowOffset = y * width;
    for (let x = 0; x < width; x++) {
      target[rowOffset + x] = sum * scale;
      sum += columnSums[columns.entering[x]] - columnSums[columns.leaving[x]];
    }
    if (y + 1 === height) {
      break;
    }
    const entering = rows.entering[y] * width;
    const leaving = rows.leaving[y] * width;
    if (squared) {
      for (let x = 0; x < width; x++) {
        const enteringValue = source[entering + x];
        const leavingValue = source[leaving + x];
        columnSums[x] += enteringValue * enteringValue - leavingValue * leavingValue;
      }
    } else {
      for (let x = 0; x < width; x++) {
        columnSums[x] += source[entering + x] - source[leaving + x];
      }
    }
  }
};

/**
 * Writes the guided filter of `signal` guided by itself to `target`. Uses `scratch.first` and `scratch.second`;
 * `target` must be neither of them nor `signal`. Radius must be at least 1.
 */
export const selfGuidedFilter = (
  signal: Float32Array,
  target: Float32Array,
  width: number,
  height: number,
  radius: number,
  epsilon: number,
  scratch: FilterScratch
): void => {
  const { columnSums, first, second } = scratch;
  const count = width * height;
  boxMean(signal, first, width, height, radius, false, columnSums);
  boxMean(signal, second, width, height, radius, true, columnSums);
  for (let i = 0; i < count; i++) {
    const mean = first[i];
    // Covariance equals the unclamped variance when the guide is the signal; only the denominator is clamped.
    const variance = second[i] - mean * mean;
    const coefficient = variance / (Math.max(variance, 0) + epsilon);
    second[i] = coefficient;
    first[i] = mean - coefficient * mean;
  }
  boxMean(second, target, width, height, radius, false, columnSums);
  boxMean(first, second, width, height, radius, false, columnSums);
  for (let i = 0; i < count; i++) {
    target[i] = target[i] * signal[i] + second[i];
  }
};

/**
 * Writes the detail between the inner- and outer-radius self-guided smooths of `signal` to `detail`. Inner radius
 * zero means the signal itself. Uses all of `scratch`; `detail` must not be one of its planes or `signal`.
 */
export const edgeAwareBand = (
  signal: Float32Array,
  detail: Float32Array,
  width: number,
  height: number,
  innerRadius: number,
  outerRadius: number,
  epsilon: number,
  scratch: FilterScratch
): void => {
  const fine = innerRadius === 0 ? signal : detail;
  if (innerRadius > 0) {
    selfGuidedFilter(signal, detail, width, height, innerRadius, epsilon, scratch);
  }
  const coarse = scratch.third;
  selfGuidedFilter(signal, coarse, width, height, outerRadius, epsilon, scratch);
  const count = width * height;
  for (let i = 0; i < count; i++) {
    detail[i] = fine[i] - coarse[i];
  }
};

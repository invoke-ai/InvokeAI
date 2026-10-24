// SPDX-License-Identifier: Apache-2.0
/**
 * Nearest-visible-pixel labels with the exact assignment the backend gets from
 * `cv2.distanceTransformWithLabels(hidden, DIST_L2, 5, labelType=DIST_LABEL_PIXEL)`. With labels OpenCV runs its
 * two-pass 5x5 chamfer transform in 16.16 fixed point rather than an exact Euclidean transform, so ties and the
 * chamfer's approximation decide which visible color a hidden pixel borrows; matching them keeps the band filters'
 * input, and so the visible result, identical to the node's.
 */

// OpenCV's L2 5x5 mask weights 1, 1.4 and 2.1969, rounded to 16.16 fixed point.
const ORTHOGONAL = 65536;
const DIAGONAL = 91750;
const KNIGHT = 143976;
/** OpenCV's initial and out-of-image distance; adding any weight to it can never win a strict comparison. */
const UNREACHED = 0xffffffff - KNIGHT;

// Forward-pass neighbours in OpenCV's comparison order. The backward pass visits their mirror images in the same
// order. A neighbour replaces the current best only when strictly closer, so ties keep the earlier one.
const NEIGHBOUR_DY = [-2, -2, -1, -1, -1, -1, -1, 0] as const;
const NEIGHBOUR_DX = [-1, 1, -2, -1, 0, 1, 2, -1] as const;
const NEIGHBOUR_WEIGHT = [KNIGHT, KNIGHT, KNIGHT, DIAGONAL, ORTHOGONAL, DIAGONAL, KNIGHT, ORTHOGONAL] as const;

/**
 * Resolves `labels` in place. On entry a visible pixel holds its own index and a hidden pixel a negative value; on
 * return every hidden pixel holds the index of the visible pixel whose color it takes. Needs at least one visible
 * pixel. `distances` is caller-owned scratch of at least `width * height` entries.
 */
export const propagateNearestVisibleLabels = (
  labels: Int32Array,
  width: number,
  height: number,
  distances: Uint32Array
): void => {
  for (let y = 0; y < height; y++) {
    const interiorRow = y >= 2;
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      if (labels[index] >= 0) {
        distances[index] = 0;
        continue;
      }
      const interior = interiorRow && x >= 2 && x < width - 2;
      let best = UNREACHED;
      let bestLabel = -1;
      for (let k = 0; k < 8; k++) {
        const nx = x + NEIGHBOUR_DX[k];
        if (!interior && (y + NEIGHBOUR_DY[k] < 0 || nx < 0 || nx >= width)) {
          continue;
        }
        const neighbour = index + NEIGHBOUR_DY[k] * width + NEIGHBOUR_DX[k];
        const distance = distances[neighbour] + NEIGHBOUR_WEIGHT[k];
        if (distance < best) {
          best = distance;
          bestLabel = labels[neighbour];
        }
      }
      distances[index] = best;
      labels[index] = bestLabel;
    }
  }

  for (let y = height - 1; y >= 0; y--) {
    const interiorRow = y < height - 2;
    for (let x = width - 1; x >= 0; x--) {
      const index = y * width + x;
      let best = distances[index];
      if (best <= ORTHOGONAL) {
        continue;
      }
      const interior = interiorRow && x >= 2 && x < width - 2;
      let bestLabel = labels[index];
      for (let k = 0; k < 8; k++) {
        const nx = x - NEIGHBOUR_DX[k];
        if (!interior && (y - NEIGHBOUR_DY[k] >= height || nx < 0 || nx >= width)) {
          continue;
        }
        const neighbour = index - NEIGHBOUR_DY[k] * width - NEIGHBOUR_DX[k];
        const distance = distances[neighbour] + NEIGHBOUR_WEIGHT[k];
        if (distance < best) {
          best = distance;
          bestLabel = labels[neighbour];
        }
      }
      distances[index] = best;
      labels[index] = bestLabel;
    }
  }
};

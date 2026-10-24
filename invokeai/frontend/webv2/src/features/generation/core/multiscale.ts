// SPDX-License-Identifier: Apache-2.0
import { guidedFilter, type ScalarImage } from './guidedFilter';

/** Returns the detail band between two guided-filter scales of the same lightness image. */
export const edgeAwareBand = (
  lightness: ScalarImage,
  innerRadius: number,
  outerRadius: number,
  epsilon: number
): ScalarImage => {
  if (!Number.isInteger(innerRadius) || innerRadius < 0) {
    throw new Error('innerRadius must be a non-negative integer');
  }
  if (!Number.isInteger(outerRadius) || outerRadius <= innerRadius) {
    throw new Error('outerRadius must be greater than innerRadius');
  }
  const inner = innerRadius === 0 ? lightness.data : guidedFilter(lightness, lightness, innerRadius, epsilon).data;
  const outer = guidedFilter(lightness, lightness, outerRadius, epsilon).data;
  const data = new Float64Array(lightness.data.length);
  for (let i = 0; i < data.length; i++) {
    data[i] = inner[i] - outer[i];
  }
  return { width: lightness.width, height: lightness.height, data };
};

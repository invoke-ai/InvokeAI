/**
 * None snaps the bbox to the model grid; auto grows undersized content toward optimal area at its aspect ratio;
 * manual uses the selected size. Restore final output to the exact bbox.
 */

import type { GenerateModelConfig, PidMode } from '@features/generation/core/types';

import { getGenerationDimensions } from '@features/generation/core/baseGenerationPolicies';
import { clampDimension, MAX_DIMENSION } from '@features/generation/core/settings';

import type { CanvasScalingSettings } from './types';

export interface CanvasSize {
  width: number;
  height: number;
}

const snap = (value: number, grid: number): number => Math.max(grid, Math.round(value / grid) * grid);

/** SDXL's trained resolutions (either orientation): already optimal, so auto scaling leaves them alone. */
const SDXL_TRAINING_DIMENSIONS: readonly (readonly [number, number])[] = [
  [512, 2048],
  [512, 1984],
  [512, 1920],
  [512, 1856],
  [576, 1792],
  [576, 1728],
  [576, 1664],
  [640, 1600],
  [640, 1536],
  [704, 1472],
  [704, 1408],
  [704, 1344],
  [768, 1344],
  [768, 1280],
  [832, 1216],
  [832, 1152],
  [896, 1152],
  [896, 1088],
  [960, 1088],
  [960, 1024],
  [1024, 1024],
];

const isSdxlTrainingSize = (base: string, { height, width }: CanvasSize): boolean =>
  base === 'sdxl' &&
  SDXL_TRAINING_DIMENSIONS.some(([a, b]) => (a === width && b === height) || (a === height && b === width));

/** Auto grows only below optimal area, preserving aspect ratio on the model grid. */
const growToOptimalArea = (size: CanvasSize, optimal: number, grid: number): CanvasSize => {
  const width = snap(size.width, grid);
  const height = snap(size.height, grid);
  const targetArea = optimal * optimal;
  if (width * height >= targetArea) {
    return { height, width };
  }
  if (width === height) {
    return { height: optimal, width: optimal };
  }
  const ratio = width / height;
  const scaled = { height, width };
  let longest = optimal - grid;
  while (scaled.width * scaled.height < targetArea && longest < MAX_DIMENSION) {
    longest += grid;
    if (ratio > 1) {
      scaled.width = longest;
      scaled.height = snap(longest / ratio, grid);
    } else {
      scaled.height = longest;
      scaled.width = snap(longest * ratio, grid);
    }
  }
  return scaled;
};

export const resolveCanvasProcessingSize = (
  // Keep the variant: Wan 5B and A14B use different grids.
  model: Pick<GenerateModelConfig, 'base' | 'type'> & { variant?: unknown },
  pidMode: PidMode,
  bbox: CanvasSize,
  scaling: CanvasScalingSettings | undefined
): CanvasSize => {
  const { grid, optimal } = getGenerationDimensions(model, pidMode);
  switch (scaling?.method) {
    case 'auto': {
      const grown = isSdxlTrainingSize(model.base, bbox) ? bbox : growToOptimalArea(bbox, optimal, grid);
      return { height: clampDimension(grown.height, grid), width: clampDimension(grown.width, grid) };
    }
    case 'manual':
      return {
        height: clampDimension(scaling.height ?? bbox.height, grid),
        width: clampDimension(scaling.width ?? bbox.width, grid),
      };
    default:
      return { height: clampDimension(bbox.height, grid), width: clampDimension(bbox.width, grid) };
  }
};

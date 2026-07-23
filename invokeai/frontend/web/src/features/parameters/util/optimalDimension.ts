import type { BaseModelType } from 'features/nodes/types/common';

/** PiD's fixed super-resolution factor (the released FLUX/SD3 checkpoints are 4x). */
export const PID_SCALE = 4;
// PiD res2k decoders are trained 512 -> 2048 (4x). In "native" mode the user-facing dimensions are the
// 4x target, so the optimal *target* dimension is 512 * 4 = 2048, regardless of the base model's own optimum.
const PID_NATIVE_OPTIMAL_DIMENSION = 512 * PID_SCALE;

/**
 * Returns the PiD generation scale that the dimension helpers should account for:
 * - 4 in "native" mode (the user-facing dimensions are the 4x target; generation runs at target / 4)
 * - 1 otherwise ('off' / 'fit' - dimensions are the generation resolution)
 */
export const getPidScale = (pidMode?: string | null): number => (pidMode === 'native' ? PID_SCALE : 1);

/**
 * Gets the optimal dimension for a given base model:
 * - sd-1, sd-2: 512
 * - sdxl, flux, sd-3, cogview4, qwen-image, z-image, anima: 1024
 * - default: 1024
 *
 * When `pidScale > 1` (PiD native mode) the user-facing dimensions are the 4x target, so the optimal is the
 * PiD target dimension (2048) instead of the model's own optimum.
 * @param base The base model
 * @param pidScale The PiD generation scale (see {@link getPidScale}); defaults to 1 (no PiD)
 * @returns The optimal dimension for the model, defaulting to 1024
 */
export const getOptimalDimension = (base?: BaseModelType | null, pidScale = 1): number => {
  if (pidScale > 1) {
    return PID_NATIVE_OPTIMAL_DIMENSION;
  }
  switch (base) {
    case 'sd-1':
    case 'sd-2':
      return 512;
    case 'sdxl':
    case 'flux':
    case 'flux2':
    case 'sd-3':
    case 'cogview4':
    case 'qwen-image':
    case 'z-image':
    case 'anima':
    default:
      return 1024;
  }
};

const SDXL_TRAINING_DIMENSIONS: [number, number][] = [
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

/**
 * Checks if the given width and height are in the SDXL training dimensions.
 * @param width The width to check
 * @param height The height to check
 * @returns Whether the width and height are in the SDXL training dimensions (order agnostic)
 */
export const isInSDXLTrainingDimensions = (width: number, height: number): boolean => {
  return SDXL_TRAINING_DIMENSIONS.some(([w, h]) => (w === width && h === height) || (w === height && h === width));
};

/**
 * Gets the grid size for a given base model. For Flux, the grid size is 16, otherwise it is 8.
 * - sd-1, sd-2, sdxl, anima: 8
 * - flux, sd-3, qwen-image, z-image: 16
 * - cogview4: 32
 * - default: 8
 * When `pidScale > 1` (PiD native mode) the grid is multiplied so the user-facing target snaps to a value
 * whose `/ pidScale` generation resolution still lands on the model's native grid.
 * @param base The base model
 * @param pidScale The PiD generation scale (see {@link getPidScale}); defaults to 1 (no PiD)
 * @returns The grid size for the model, defaulting to 8
 */
export const getGridSize = (base?: BaseModelType | null, pidScale = 1): number => {
  let gridSize: number;
  switch (base) {
    case 'cogview4':
      gridSize = 32;
      break;
    case 'flux':
    case 'flux2':
    case 'sd-3':
    case 'qwen-image':
    case 'z-image':
      gridSize = 16;
      break;
    case 'sd-1':
    case 'sd-2':
    case 'sdxl':
    case 'anima':
    default:
      gridSize = 8;
      break;
  }
  return gridSize * pidScale;
};

const MIN_AREA_FACTOR = 0.8;
const MAX_AREA_FACTOR = 1.2;

export const getIsSizeTooSmall = (width: number, height: number, optimalDimension: number): boolean => {
  const currentArea = width * height;
  const optimalArea = optimalDimension * optimalDimension;
  if (currentArea < optimalArea * MIN_AREA_FACTOR) {
    return true;
  }
  return false;
};

export const getIsSizeTooLarge = (width: number, height: number, optimalDimension: number): boolean => {
  const currentArea = width * height;
  const optimalArea = optimalDimension * optimalDimension;
  if (currentArea > optimalArea * MAX_AREA_FACTOR) {
    return true;
  }
  return false;
};

/**
 * Gets whether the current width and height needs to be resized to the optimal dimension.
 * The current width and height needs to be resized if the current area is not within 20% of the optimal area.
 * @param width The width to compare with the optimal dimension
 * @param height The height to compare with the optimal dimension
 * @param optimalDimension The optimal dimension
 * @returns Whether the current width and height needs to be resized to the optimal dimension
 */
export const getIsSizeOptimal = (width: number, height: number, base?: BaseModelType, pidScale = 1): boolean => {
  const optimalDimension = getOptimalDimension(base, pidScale);
  return !getIsSizeTooSmall(width, height, optimalDimension) && !getIsSizeTooLarge(width, height, optimalDimension);
};

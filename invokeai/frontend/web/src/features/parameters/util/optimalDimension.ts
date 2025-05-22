import type { BaseModelType } from 'services/api/types';

/**
 * Gets the optimal dimension for a given base model:
 * - sd-1, sd-2: 512
 * - sdxl, flux, sd-3, cogview4: 1024
 * - default: 1024
 * @param base The base model
 * @returns The optimal dimension for the model, defaulting to 1024
 */
export const getOptimalDimension = (base?: BaseModelType | null): number => {
  switch (base) {
    case 'sd-1':
    case 'sd-2':
      return 512;
    case 'sdxl':
    case 'flux':
    case 'sd-3':
    case 'cogview4':
    case 'imagen3':
    case 'imagen4':
    case 'chatgpt-4o':
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
 * - sd-1, sd-2, sdxl: 8
 * - flux, sd-3: 16
 * - cogview4: 32
 * - default: 8
 * @param base The base model
 * @returns The grid size for the model, defaulting to 8
 */
export const getGridSize = (base?: BaseModelType | null): number => {
  switch (base) {
    case 'cogview4':
      return 32;
    case 'flux':
    case 'sd-3':
      return 16;
    case 'sd-1':
    case 'sd-2':
    case 'sdxl':
    case 'imagen3':
    case 'chatgpt-4o':
    default:
      return 8;
  }
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
export const getIsSizeOptimal = (width: number, height: number, base?: BaseModelType): boolean => {
  const optimalDimension = getOptimalDimension(base);
  return !getIsSizeTooSmall(width, height, optimalDimension) && !getIsSizeTooLarge(width, height, optimalDimension);
};

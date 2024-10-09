import type { MainModelBase } from 'features/nodes/types/common';
import type { BaseModelType } from 'services/api/types';

/**
 * Gets the optimal dimension for a given base model:
 * - sd-1, sd-2: 512
 * - sdxl, flux: 1024
 * - default: 1024
 * @param base The base model
 * @returns The optimal dimension for the model, defaulting to 512
 */
export const getOptimalDimension = (base?: BaseModelType | null): number => {
  switch (base) {
    case 'sd-1':
    case 'sd-2':
      return 512;
    case 'sdxl':
    case 'flux':
    default:
      return 1024;
  }
};

/**
 * Gets the grid size for a given base model. For Flux, the grid size is 16, otherwise it is 8.
 * - sd-1, sd-2, sdxl: 8
 * - flux: 16
 * - default: 8
 * @param base The base model
 * @returns The grid size for the model, defaulting to 8
 */
export const getGridSize = (base?: BaseModelType | null): number => {
  switch (base) {
    case 'flux':
      return 16;
    case 'sd-1':
    case 'sd-2':
    case 'sdxl':
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
export const getIsSizeOptimal = (width: number, height: number, modelBase: MainModelBase): boolean => {
  const optimalDimension = getOptimalDimension(modelBase);
  return !getIsSizeTooSmall(width, height, optimalDimension) && !getIsSizeTooLarge(width, height, optimalDimension);
};

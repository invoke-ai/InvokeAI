import type { ModelIdentifierField } from 'features/nodes/types/common';

/**
 * Gets the optimal dimension for a givel model, based on the model's base_model
 * @param model The model identifier
 * @returns The optimal dimension for the model
 */
export const getOptimalDimension = (model?: ModelIdentifierField | null): number =>
  model?.base === 'sdxl' ? 1024 : 512;

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
export const getIsSizeOptimal = (width: number, height: number, optimalDimension: number): boolean => {
  return !getIsSizeTooSmall(width, height, optimalDimension) && !getIsSizeTooLarge(width, height, optimalDimension);
};

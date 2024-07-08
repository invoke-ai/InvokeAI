import { roundToMultiple } from 'common/util/roundDownToMultiple';

/**
 * Calculate the new width and height that will fit the given aspect ratio, retaining the input area
 * @param ratio The aspect ratio to calculate the new size for
 * @param area The input area
 * @returns The width and height that will fit the given aspect ratio, retaining the input area
 */
export const calculateNewSize = (ratio: number, area: number): { width: number; height: number } => {
  const exactWidth = Math.sqrt(area * ratio);
  const exactHeight = exactWidth / ratio;

  return {
    width: roundToMultiple(exactWidth, 8),
    height: roundToMultiple(exactHeight, 8),
  };
};

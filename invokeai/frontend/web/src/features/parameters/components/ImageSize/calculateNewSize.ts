import { roundToMultiple } from 'common/util/roundDownToMultiple';

/**
 * Calculate the new width and height that will fit the given aspect ratio, retaining the input area
 * @param ratio The aspect ratio to calculate the new size for
 * @param width The input width
 * @param height The input height
 * @returns The width and height that will fit the given aspect ratio, retaining the input area
 */
export const calculateNewSize = (
  ratio: number,
  width: number,
  height: number
): { width: number; height: number } => {
  const area = width * height;
  const newWidth = roundToMultiple(Math.sqrt(area * ratio), 8);
  const newHeight = roundToMultiple(area / newWidth, 8);
  return { width: newWidth, height: newHeight };
};

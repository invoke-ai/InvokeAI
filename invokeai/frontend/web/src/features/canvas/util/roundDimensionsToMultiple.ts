import { roundToMultiple } from 'common/util/roundDownToMultiple';
import type { Dimensions } from 'features/canvas/store/canvasTypes';

const roundDimensionsToMultiple = (dimensions: Dimensions, multiple: number): Dimensions => {
  return {
    width: roundToMultiple(dimensions.width, multiple),
    height: roundToMultiple(dimensions.height, multiple),
  };
};

export default roundDimensionsToMultiple;

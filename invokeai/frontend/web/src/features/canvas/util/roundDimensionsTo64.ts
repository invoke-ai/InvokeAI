import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { Dimensions } from '../store/canvasTypes';

const roundDimensionsTo64 = (dimensions: Dimensions): Dimensions => {
  return {
    width: roundToMultiple(dimensions.width, 64),
    height: roundToMultiple(dimensions.height, 64),
  };
};

export default roundDimensionsTo64;

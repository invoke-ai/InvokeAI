import { FormLabel } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAINumberInput from '../../../../common/components/IAINumberInput';
import IAISlider from '../../../../common/components/IAISlider';
import { roundDownToMultiple } from '../../../../common/util/roundDownToMultiple';
import {
  InpaintingState,
  setBoundingBoxDimensions,
} from '../../../tabs/Inpainting/inpaintingSlice';

const boundingBoxDimensionsSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const { canvasDimensions, boundingBoxDimensions } = inpainting;
    return { canvasDimensions, boundingBoxDimensions };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const BoundingBoxDimensions = () => {
  const dispatch = useAppDispatch();
  const { canvasDimensions, boundingBoxDimensions } = useAppSelector(
    boundingBoxDimensionsSelector
  );

  const handleChangeBoundingBoxWidth = (v: number) => {
    dispatch(setBoundingBoxDimensions({ ...boundingBoxDimensions, width: v }));
  };

  const handleChangeBoundingBoxHeight = (v: number) => {
    dispatch(setBoundingBoxDimensions({ ...boundingBoxDimensions, height: v }));
  };

  return (
    <div className="inpainting-bounding-box-dimensions">
      Inpainting Bounding Box
      <div className="inpainting-bounding-box-dimensions-slider-numberinput">
        <IAISlider
          label="Width"
          width={'8rem'}
          min={64}
          max={roundDownToMultiple(canvasDimensions.width, 64)}
          step={64}
          value={boundingBoxDimensions.width}
          onChange={handleChangeBoundingBoxWidth}
        />
        <IAINumberInput
          value={boundingBoxDimensions.width}
          onChange={handleChangeBoundingBoxWidth}
          min={64}
          max={roundDownToMultiple(canvasDimensions.width, 64)}
          step={64}
          width={'5.5rem'}
        />
      </div>
      <div className="inpainting-bounding-box-dimensions-slider-numberinput">
        <IAISlider
          label="Height"
          width={'8rem'}
          min={64}
          max={roundDownToMultiple(canvasDimensions.height, 64)}
          step={64}
          value={boundingBoxDimensions.height}
          onChange={handleChangeBoundingBoxHeight}
        />
        <IAINumberInput
          value={boundingBoxDimensions.height}
          onChange={handleChangeBoundingBoxHeight}
          min={64}
          max={roundDownToMultiple(canvasDimensions.height, 64)}
          step={64}
          width={'5.5rem'}
        />
      </div>
    </div>
  );
};

export default BoundingBoxDimensions;

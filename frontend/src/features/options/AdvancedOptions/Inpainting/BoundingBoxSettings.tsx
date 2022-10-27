import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { ChangeEvent } from 'react';
import { BiReset } from 'react-icons/bi';

import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAICheckbox from '../../../../common/components/IAICheckbox';
import IAIIconButton from '../../../../common/components/IAIIconButton';

import IAINumberInput from '../../../../common/components/IAINumberInput';
import IAISlider from '../../../../common/components/IAISlider';
import IAISwitch from '../../../../common/components/IAISwitch';
import { roundDownToMultiple } from '../../../../common/util/roundDownToMultiple';
import {
  InpaintingState,
  setBoundingBoxDimensions,
  setShouldShowBoundingBox,
  setShouldShowBoundingBoxFill,
} from '../../../tabs/Inpainting/inpaintingSlice';

const boundingBoxDimensionsSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const {
      canvasDimensions,
      boundingBoxDimensions,
      shouldShowBoundingBox,
      shouldShowBoundingBoxFill,
      pastLines,
      futureLines,
    } = inpainting;
    return {
      canvasDimensions,
      boundingBoxDimensions,
      shouldShowBoundingBox,
      shouldShowBoundingBoxFill,
      pastLines,
      futureLines,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const BoundingBoxSettings = () => {
  const dispatch = useAppDispatch();
  const {
    canvasDimensions,
    boundingBoxDimensions,
    shouldShowBoundingBox,
    shouldShowBoundingBoxFill,
  } = useAppSelector(boundingBoxDimensionsSelector);

  const handleChangeBoundingBoxWidth = (v: number) => {
    dispatch(setBoundingBoxDimensions({ ...boundingBoxDimensions, width: v }));
  };

  const handleChangeBoundingBoxHeight = (v: number) => {
    dispatch(setBoundingBoxDimensions({ ...boundingBoxDimensions, height: v }));
  };

  const handleShowBoundingBox = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldShowBoundingBox(e.target.checked));

  const handleChangeShouldShowBoundingBoxFill = () => {
    dispatch(setShouldShowBoundingBoxFill(!shouldShowBoundingBoxFill));
  };

  const handleResetWidth = () => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        width: canvasDimensions.width,
      })
    );
  };

  const handleResetHeight = () => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        height: canvasDimensions.height,
      })
    );
  };

  return (
    <div className="inpainting-bounding-box-settings">
      <div className="inpainting-bounding-box-header">
        <p>Inpaint Box</p>
        <IAISwitch
          isChecked={shouldShowBoundingBox}
          width={'auto'}
          onChange={handleShowBoundingBox}
        />
      </div>
      <div className="inpainting-bounding-box-settings-items">
        <div className="inpainting-bounding-box-dimensions-slider-numberinput">
          <IAISlider
            label="Box W"
            min={64}
            max={roundDownToMultiple(canvasDimensions.width, 64)}
            step={64}
            value={boundingBoxDimensions.width}
            onChange={handleChangeBoundingBoxWidth}
            isDisabled={!shouldShowBoundingBox}
            width={'5rem'}
          />
          <IAINumberInput
            value={boundingBoxDimensions.width}
            onChange={handleChangeBoundingBoxWidth}
            min={64}
            max={roundDownToMultiple(canvasDimensions.width, 64)}
            step={64}
            isDisabled={!shouldShowBoundingBox}
            width={'5rem'}
          />
          <IAIIconButton
            size={'sm'}
            aria-label={'Reset Width'}
            tooltip={'Reset Width'}
            onClick={handleResetWidth}
            icon={<BiReset />}
            styleClass="inpainting-bounding-box-reset-icon-btn"
          />
        </div>
        <div className="inpainting-bounding-box-dimensions-slider-numberinput">
          <IAISlider
            label="Box H"
            min={64}
            max={roundDownToMultiple(canvasDimensions.height, 64)}
            step={64}
            value={boundingBoxDimensions.height}
            onChange={handleChangeBoundingBoxHeight}
            isDisabled={!shouldShowBoundingBox}
            width={'5rem'}
          />
          <IAINumberInput
            value={boundingBoxDimensions.height}
            onChange={handleChangeBoundingBoxHeight}
            min={64}
            max={roundDownToMultiple(canvasDimensions.height, 64)}
            step={64}
            padding="0"
            isDisabled={!shouldShowBoundingBox}
            width={'5rem'}
          />
          <IAIIconButton
            size={'sm'}
            aria-label={'Reset Height'}
            tooltip={'Reset Height'}
            onClick={handleResetHeight}
            icon={<BiReset />}
            styleClass="inpainting-bounding-box-reset-icon-btn"
          />
        </div>
        <IAICheckbox
          label="Darken Outside Box"
          isChecked={shouldShowBoundingBoxFill}
          onChange={handleChangeShouldShowBoundingBoxFill}
          styleClass="inpainting-bounding-box-darken"
          isDisabled={!shouldShowBoundingBox}
        />
      </div>
    </div>
  );
};

export default BoundingBoxSettings;

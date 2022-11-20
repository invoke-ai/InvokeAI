import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAISlider from 'common/components/IAISlider';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { setBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import _ from 'lodash';

const selector = createSelector(
  canvasSelector,
  (canvas) => {
    const { boundingBoxDimensions } = canvas;
    return {
      boundingBoxDimensions,
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
  const { boundingBoxDimensions } = useAppSelector(selector);

  const handleChangeWidth = (v: number) => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        width: Math.floor(v),
      })
    );
  };

  const handleChangeHeight = (v: number) => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        height: Math.floor(v),
      })
    );
  };

  const handleResetWidth = () => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        width: Math.floor(512),
      })
    );
  };

  const handleResetHeight = () => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        height: Math.floor(512),
      })
    );
  };

  return (
    <div className="inpainting-bounding-box-settings">
      <div className="inpainting-bounding-box-header">
        <p>Canvas Bounding Box</p>
      </div>
      <div className="inpainting-bounding-box-settings-items">
        <IAISlider
          label={'Width'}
          min={64}
          max={1024}
          step={64}
          value={boundingBoxDimensions.width}
          onChange={handleChangeWidth}
          handleReset={handleResetWidth}
          sliderNumberInputProps={{ max: 4096 }}
          withSliderMarks
          withInput
          withReset
        />
        <IAISlider
          label={'Height'}
          min={64}
          max={1024}
          step={64}
          value={boundingBoxDimensions.height}
          onChange={handleChangeHeight}
          handleReset={handleResetHeight}
          sliderNumberInputProps={{ max: 4096 }}
          withSliderMarks
          withInput
          withReset
        />
      </div>
    </div>
  );
};

export default BoundingBoxSettings;

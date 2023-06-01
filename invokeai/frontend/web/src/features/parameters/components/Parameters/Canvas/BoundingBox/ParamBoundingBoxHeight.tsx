import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import { setBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { memo } from 'react';

import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [canvasSelector, isStagingSelector],
  (canvas, isStaging) => {
    const { boundingBoxDimensions } = canvas;
    return {
      boundingBoxDimensions,
      isStaging,
    };
  },
  defaultSelectorOptions
);

const ParamBoundingBoxWidth = () => {
  const dispatch = useAppDispatch();
  const { boundingBoxDimensions, isStaging } = useAppSelector(selector);

  const { t } = useTranslation();

  const handleChangeHeight = (v: number) => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        height: Math.floor(v),
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
    <IAISlider
      label={t('parameters.boundingBoxHeight')}
      min={64}
      max={1024}
      step={64}
      value={boundingBoxDimensions.height}
      onChange={handleChangeHeight}
      isDisabled={isStaging}
      sliderNumberInputProps={{ max: 4096 }}
      withSliderMarks
      withInput
      inputReadOnly
      withReset
      handleReset={handleResetHeight}
    />
  );
};

export default memo(ParamBoundingBoxWidth);

import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { setBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { memo } from 'react';

import { useTranslation } from 'react-i18next';

const selector = createSelector(
  canvasSelector,
  (canvas) => {
    const { boundingBoxDimensions } = canvas;
    return {
      boundingBoxDimensions,
    };
  },
  defaultSelectorOptions
);

const ParamBoundingBoxWidth = () => {
  const dispatch = useAppDispatch();
  const { boundingBoxDimensions } = useAppSelector(selector);

  const { t } = useTranslation();

  const handleChangeWidth = (v: number) => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        width: Math.floor(v),
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

  return (
    <IAISlider
      label={t('parameters.width')}
      min={64}
      max={1024}
      step={64}
      value={boundingBoxDimensions.width}
      onChange={handleChangeWidth}
      sliderNumberInputProps={{ max: 4096 }}
      withSliderMarks
      withInput
      inputReadOnly
      withReset
      handleReset={handleResetWidth}
    />
  );
};

export default memo(ParamBoundingBoxWidth);

import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { setScaledBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [generationSelector, systemSelector, canvasSelector],
  (parameters, system, canvas) => {
    const { scaledBoundingBoxDimensions, boundingBoxScaleMethod } = canvas;

    return {
      scaledBoundingBoxDimensions,
      isManual: boundingBoxScaleMethod === 'manual',
    };
  },
  defaultSelectorOptions
);

const ParamScaledHeight = () => {
  const dispatch = useAppDispatch();
  const { isManual, scaledBoundingBoxDimensions } = useAppSelector(selector);

  const { t } = useTranslation();

  const handleChangeScaledHeight = (v: number) => {
    dispatch(
      setScaledBoundingBoxDimensions({
        ...scaledBoundingBoxDimensions,
        height: Math.floor(v),
      })
    );
  };

  const handleResetScaledHeight = () => {
    dispatch(
      setScaledBoundingBoxDimensions({
        ...scaledBoundingBoxDimensions,
        height: Math.floor(512),
      })
    );
  };

  return (
    <IAISlider
      isDisabled={!isManual}
      label={t('parameters.scaledHeight')}
      min={64}
      max={1024}
      step={64}
      value={scaledBoundingBoxDimensions.height}
      onChange={handleChangeScaledHeight}
      sliderNumberInputProps={{ max: 4096 }}
      withSliderMarks
      withInput
      inputReadOnly
      withReset
      handleReset={handleResetScaledHeight}
    />
  );
};

export default memo(ParamScaledHeight);

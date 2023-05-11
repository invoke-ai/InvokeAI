import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { setScaledBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const { boundingBoxScaleMethod, scaledBoundingBoxDimensions } = canvas;

    return {
      scaledBoundingBoxDimensions,
      isManual: boundingBoxScaleMethod === 'manual',
    };
  },
  defaultSelectorOptions
);

const ParamScaledWidth = () => {
  const dispatch = useAppDispatch();
  const { isManual, scaledBoundingBoxDimensions } = useAppSelector(selector);

  const { t } = useTranslation();

  const handleChangeScaledWidth = (v: number) => {
    dispatch(
      setScaledBoundingBoxDimensions({
        ...scaledBoundingBoxDimensions,
        width: Math.floor(v),
      })
    );
  };

  const handleResetScaledWidth = () => {
    dispatch(
      setScaledBoundingBoxDimensions({
        ...scaledBoundingBoxDimensions,
        width: Math.floor(512),
      })
    );
  };

  return (
    <IAISlider
      isDisabled={!isManual}
      label={t('parameters.scaledWidth')}
      min={64}
      max={1024}
      step={64}
      value={scaledBoundingBoxDimensions.width}
      onChange={handleChangeScaledWidth}
      sliderNumberInputProps={{ max: 4096 }}
      withSliderMarks
      withInput
      inputReadOnly
      withReset
      handleReset={handleResetScaledWidth}
    />
  );
};

export default memo(ParamScaledWidth);

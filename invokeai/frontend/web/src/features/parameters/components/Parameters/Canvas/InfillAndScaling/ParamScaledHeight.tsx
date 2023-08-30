import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { setScaledBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [generationSelector, canvasSelector],
  (generation, canvas) => {
    const { scaledBoundingBoxDimensions, boundingBoxScaleMethod } = canvas;
    const { model } = generation;

    return {
      model,
      scaledBoundingBoxDimensions,
      isManual: boundingBoxScaleMethod === 'manual',
    };
  },
  defaultSelectorOptions
);

const ParamScaledHeight = () => {
  const dispatch = useAppDispatch();
  const { model, isManual, scaledBoundingBoxDimensions } =
    useAppSelector(selector);

  const initial = ['sdxl', 'sdxl-refiner'].includes(model?.base_model as string)
    ? 1024
    : 512;

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
        height: Math.floor(initial),
      })
    );
  };

  return (
    <IAISlider
      isDisabled={!isManual}
      label={t('parameters.scaledHeight')}
      min={64}
      max={1536}
      step={64}
      value={scaledBoundingBoxDimensions.height}
      onChange={handleChangeScaledHeight}
      sliderNumberInputProps={{ max: 4096 }}
      withSliderMarks
      withInput
      withReset
      handleReset={handleResetScaledHeight}
    />
  );
};

export default memo(ParamScaledHeight);

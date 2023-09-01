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
  [canvasSelector, generationSelector],
  (canvas, generation) => {
    const { boundingBoxScaleMethod, scaledBoundingBoxDimensions } = canvas;
    const { model } = generation;

    return {
      model,
      scaledBoundingBoxDimensions,
      isManual: boundingBoxScaleMethod === 'manual',
    };
  },
  defaultSelectorOptions
);

const ParamScaledWidth = () => {
  const dispatch = useAppDispatch();
  const { model, isManual, scaledBoundingBoxDimensions } =
    useAppSelector(selector);

  const initial = ['sdxl', 'sdxl-refiner'].includes(model?.base_model as string)
    ? 1024
    : 512;

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
        width: Math.floor(initial),
      })
    );
  };

  return (
    <IAISlider
      isDisabled={!isManual}
      label={t('parameters.scaledWidth')}
      min={64}
      max={1536}
      step={64}
      value={scaledBoundingBoxDimensions.width}
      onChange={handleChangeScaledWidth}
      sliderNumberInputProps={{ max: 4096 }}
      withSliderMarks
      withInput
      withReset
      handleReset={handleResetScaledWidth}
    />
  );
};

export default memo(ParamScaledWidth);

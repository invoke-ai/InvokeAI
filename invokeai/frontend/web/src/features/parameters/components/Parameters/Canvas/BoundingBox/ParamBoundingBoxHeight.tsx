import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { setBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { memo } from 'react';

import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [stateSelector, isStagingSelector],
  ({ canvas, generation }, isStaging) => {
    const { boundingBoxDimensions } = canvas;
    const { model, aspectRatio } = generation;
    return {
      model,
      boundingBoxDimensions,
      isStaging,
      aspectRatio,
    };
  },
  defaultSelectorOptions
);

const ParamBoundingBoxWidth = () => {
  const dispatch = useAppDispatch();
  const { model, boundingBoxDimensions, isStaging, aspectRatio } =
    useAppSelector(selector);

  const { t } = useTranslation();

  const initial = ['sdxl', 'sdxl-refiner'].includes(model?.base_model as string)
    ? 1024
    : 512;

  const handleChangeHeight = (v: number) => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        height: Math.floor(v),
      })
    );
    if (aspectRatio) {
      const newWidth = roundToMultiple(v * aspectRatio, 64);
      dispatch(
        setBoundingBoxDimensions({
          width: newWidth,
          height: Math.floor(v),
        })
      );
    }
  };

  const handleResetHeight = () => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        height: Math.floor(initial),
      })
    );
    if (aspectRatio) {
      const newWidth = roundToMultiple(initial * aspectRatio, 64);
      dispatch(
        setBoundingBoxDimensions({
          width: newWidth,
          height: Math.floor(initial),
        })
      );
    }
  };

  return (
    <IAISlider
      label={t('parameters.boundingBoxHeight')}
      min={64}
      max={1536}
      step={64}
      value={boundingBoxDimensions.height}
      onChange={handleChangeHeight}
      isDisabled={isStaging}
      sliderNumberInputProps={{ max: 4096 }}
      withSliderMarks
      withInput
      withReset
      handleReset={handleResetHeight}
    />
  );
};

export default memo(ParamBoundingBoxWidth);

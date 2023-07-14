import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import { setBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { memo } from 'react';

import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [canvasSelector, isStagingSelector, uiSelector],
  (canvas, isStaging, ui) => {
    const { boundingBoxDimensions } = canvas;
    const { aspectRatio } = ui;
    return {
      boundingBoxDimensions,
      isStaging,
      aspectRatio,
    };
  },
  defaultSelectorOptions
);

const ParamBoundingBoxWidth = () => {
  const dispatch = useAppDispatch();
  const { boundingBoxDimensions, isStaging, aspectRatio } =
    useAppSelector(selector);

  const { t } = useTranslation();

  const handleChangeWidth = (v: number) => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        width: Math.floor(v),
      })
    );
    if (aspectRatio) {
      const newHeight = roundToMultiple(v / aspectRatio, 64);
      dispatch(
        setBoundingBoxDimensions({
          width: Math.floor(v),
          height: newHeight,
        })
      );
    }
  };

  const handleResetWidth = () => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        width: Math.floor(512),
      })
    );
    if (aspectRatio) {
      const newHeight = roundToMultiple(512 / aspectRatio, 64);
      dispatch(
        setBoundingBoxDimensions({
          width: Math.floor(512),
          height: newHeight,
        })
      );
    }
  };

  return (
    <IAISlider
      label={t('parameters.boundingBoxWidth')}
      min={64}
      max={1024}
      step={64}
      value={boundingBoxDimensions.width}
      onChange={handleChangeWidth}
      isDisabled={isStaging}
      sliderNumberInputProps={{ max: 4096 }}
      withSliderMarks
      withInput
      withReset
      handleReset={handleResetWidth}
    />
  );
};

export default memo(ParamBoundingBoxWidth);

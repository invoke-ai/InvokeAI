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
        height: Math.floor(512),
      })
    );
    if (aspectRatio) {
      const newWidth = roundToMultiple(512 * aspectRatio, 64);
      dispatch(
        setBoundingBoxDimensions({
          width: newWidth,
          height: Math.floor(512),
        })
      );
    }
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
      withReset
      handleReset={handleResetHeight}
    />
  );
};

export default memo(ParamBoundingBoxWidth);

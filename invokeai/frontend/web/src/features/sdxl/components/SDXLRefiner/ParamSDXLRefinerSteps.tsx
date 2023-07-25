import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAINumberInput from 'common/components/IAINumberInput';

import IAISlider from 'common/components/IAISlider';
import { clampSymmetrySteps } from 'features/parameters/store/generationSlice';
import { setRefinerSteps } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [stateSelector],
  ({ sdxl, ui }) => {
    const { refinerSteps } = sdxl;
    const { shouldUseSliders } = ui;

    return {
      refinerSteps,

      shouldUseSliders,
    };
  },
  defaultSelectorOptions
);

const ParamSDXLRefinerSteps = () => {
  const { refinerSteps, shouldUseSliders } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setRefinerSteps(v));
    },
    [dispatch]
  );
  const handleReset = useCallback(() => {
    dispatch(setRefinerSteps(20));
  }, [dispatch]);

  const handleBlur = useCallback(() => {
    dispatch(clampSymmetrySteps());
  }, [dispatch]);

  return shouldUseSliders ? (
    <IAISlider
      label={t('parameters.steps')}
      min={1}
      max={200}
      step={1}
      onChange={handleChange}
      handleReset={handleReset}
      value={refinerSteps}
      withInput
      withReset
      withSliderMarks
      sliderNumberInputProps={{ max: 999 }}
    />
  ) : (
    <IAINumberInput
      label={t('parameters.steps')}
      min={1}
      max={999}
      step={1}
      onChange={handleChange}
      value={refinerSteps}
      numberInputFieldProps={{ textAlign: 'center' }}
      onBlur={handleBlur}
    />
  );
};

export default memo(ParamSDXLRefinerSteps);

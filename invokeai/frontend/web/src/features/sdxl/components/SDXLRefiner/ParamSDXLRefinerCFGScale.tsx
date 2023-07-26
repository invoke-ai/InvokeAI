import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAINumberInput from 'common/components/IAINumberInput';
import IAISlider from 'common/components/IAISlider';
import { setRefinerCFGScale } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useIsRefinerAvailable } from 'services/api/hooks/useIsRefinerAvailable';

const selector = createSelector(
  [stateSelector],
  ({ sdxl, ui, hotkeys }) => {
    const { refinerCFGScale } = sdxl;
    const { shouldUseSliders } = ui;
    const { shift } = hotkeys;

    return {
      refinerCFGScale,
      shouldUseSliders,
      shift,
    };
  },
  defaultSelectorOptions
);

const ParamSDXLRefinerCFGScale = () => {
  const { refinerCFGScale, shouldUseSliders, shift } = useAppSelector(selector);
  const isRefinerAvailable = useIsRefinerAvailable();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => dispatch(setRefinerCFGScale(v)),
    [dispatch]
  );

  const handleReset = useCallback(
    () => dispatch(setRefinerCFGScale(7)),
    [dispatch]
  );

  return shouldUseSliders ? (
    <IAISlider
      label={t('parameters.cfgScale')}
      step={shift ? 0.1 : 0.5}
      min={1}
      max={20}
      onChange={handleChange}
      handleReset={handleReset}
      value={refinerCFGScale}
      sliderNumberInputProps={{ max: 200 }}
      withInput
      withReset
      withSliderMarks
      isInteger={false}
      isDisabled={!isRefinerAvailable}
    />
  ) : (
    <IAINumberInput
      label={t('parameters.cfgScale')}
      step={0.5}
      min={1}
      max={200}
      onChange={handleChange}
      value={refinerCFGScale}
      isInteger={false}
      numberInputFieldProps={{ textAlign: 'center' }}
      isDisabled={!isRefinerAvailable}
    />
  );
};

export default memo(ParamSDXLRefinerCFGScale);

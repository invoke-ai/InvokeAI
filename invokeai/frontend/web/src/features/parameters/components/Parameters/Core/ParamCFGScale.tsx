import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import IAISlider from 'common/components/IAISlider';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setCfgScale } from 'features/parameters/store/generationSlice';
import { configSelector } from 'features/system/store/configSelectors';
import { hotkeysSelector } from 'features/ui/store/hotkeysSlice';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [generationSelector, configSelector, uiSelector, hotkeysSelector],
  (generation, config, ui, hotkeys) => {
    const { initial, min, sliderMax, inputMax } = config.sd.guidance;
    const { cfgScale } = generation;
    const { shouldUseSliders } = ui;
    const { shift } = hotkeys;

    return {
      cfgScale,
      initial,
      min,
      sliderMax,
      inputMax,
      shouldUseSliders,
      shift,
    };
  }
);

const ParamCFGScale = () => {
  const {
    cfgScale,
    initial,
    min,
    sliderMax,
    inputMax,
    shouldUseSliders,
    shift,
  } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => dispatch(setCfgScale(v)),
    [dispatch]
  );

  const handleReset = useCallback(
    () => dispatch(setCfgScale(initial)),
    [dispatch, initial]
  );

  return shouldUseSliders ? (
    <IAISlider
      label={t('parameters.cfgScale')}
      step={shift ? 0.1 : 0.5}
      min={min}
      max={sliderMax}
      onChange={handleChange}
      handleReset={handleReset}
      value={cfgScale}
      sliderNumberInputProps={{ max: inputMax }}
      withInput
      withReset
      withSliderMarks
      isInteger={false}
    />
  ) : (
    <IAINumberInput
      label={t('parameters.cfgScale')}
      step={0.5}
      min={min}
      max={inputMax}
      onChange={handleChange}
      value={cfgScale}
      isInteger={false}
      numberInputFieldProps={{ textAlign: 'center' }}
    />
  );
};

export default memo(ParamCFGScale);

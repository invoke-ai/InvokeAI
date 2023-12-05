import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAISlider from 'common/components/IAISlider';
import { setCfgRescaleMultiplier } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [stateSelector],
  ({ generation, hotkeys }) => {
    const { cfgRescaleMultiplier } = generation;
    const { shift } = hotkeys;

    return {
      cfgRescaleMultiplier,
      shift,
    };
  },
  defaultSelectorOptions
);

const ParamCFGRescaleMultiplier = () => {
  const { cfgRescaleMultiplier, shift } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => dispatch(setCfgRescaleMultiplier(v)),
    [dispatch]
  );

  const handleReset = useCallback(
    () => dispatch(setCfgRescaleMultiplier(0)),
    [dispatch]
  );

  return (
    <IAIInformationalPopover feature="paramCFGRescaleMultiplier">
      <IAISlider
        label={t('parameters.cfgRescaleMultiplier')}
        step={shift ? 0.01 : 0.05}
        min={0}
        max={0.99}
        onChange={handleChange}
        handleReset={handleReset}
        value={cfgRescaleMultiplier}
        sliderNumberInputProps={{ max: 0.99 }}
        withInput
        withReset
        withSliderMarks
        isInteger={false}
      />
    </IAIInformationalPopover>
  );
};

export default memo(ParamCFGRescaleMultiplier);

import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISwitch from 'common/components/IAISwitch';
import { NoiseUseCPUPopover } from 'features/informationalPopovers/components/noiseUseCPU';
import { shouldUseCpuNoiseChanged } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { shouldUseNoiseSettings, shouldUseCpuNoise } = state.generation;
    return {
      isDisabled: !shouldUseNoiseSettings,
      shouldUseCpuNoise,
    };
  },
  defaultSelectorOptions
);

export const ParamCpuNoiseToggle = () => {
  const dispatch = useAppDispatch();
  const { isDisabled, shouldUseCpuNoise } = useAppSelector(selector);
  const { t } = useTranslation();

  const handleChange = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(shouldUseCpuNoiseChanged(e.target.checked));

  return (
    <NoiseUseCPUPopover>
      <IAISwitch
        isDisabled={isDisabled}
        label={t('parameters.useCpuNoise')}
        isChecked={shouldUseCpuNoise}
        onChange={handleChange}
      />
    </NoiseUseCPUPopover>
  );
};

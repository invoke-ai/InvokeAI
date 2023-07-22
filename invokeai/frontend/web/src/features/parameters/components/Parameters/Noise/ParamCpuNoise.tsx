import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISwitch from 'common/components/IAISwitch';
import { shouldUseCpuNoiseChanged } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';

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

  const handleChange = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(shouldUseCpuNoiseChanged(e.target.checked));

  return (
    <IAISwitch
      isDisabled={isDisabled}
      label="Use CPU Noise"
      isChecked={shouldUseCpuNoise}
      onChange={handleChange}
    />
  );
};

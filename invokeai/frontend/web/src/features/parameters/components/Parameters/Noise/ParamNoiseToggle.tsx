import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldUseNoiseSettings } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';

export const ParamNoiseToggle = () => {
  const dispatch = useAppDispatch();

  const shouldUseNoiseSettings = useAppSelector(
    (state: RootState) => state.generation.shouldUseNoiseSettings
  );

  const handleChange = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldUseNoiseSettings(e.target.checked));

  return (
    <IAISwitch
      label="Enable Noise Settings"
      isChecked={shouldUseNoiseSettings}
      onChange={handleChange}
    />
  );
};

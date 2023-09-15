import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldUseNoiseSettings } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamNoiseToggle = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const shouldUseNoiseSettings = useAppSelector(
    (state: RootState) => state.generation.shouldUseNoiseSettings
  );

  const handleChange = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldUseNoiseSettings(e.target.checked));

  return (
    <IAIInformationalPopover details="noiseEnable">
      <IAISwitch
        label={t('parameters.enableNoiseSettings')}
        isChecked={shouldUseNoiseSettings}
        onChange={handleChange}
      />
    </IAIInformationalPopover>
  );
};

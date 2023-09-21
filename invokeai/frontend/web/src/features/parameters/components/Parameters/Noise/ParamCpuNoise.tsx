import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover';
import IAISwitch from 'common/components/IAISwitch';
import { shouldUseCpuNoiseChanged } from 'features/parameters/store/generationSlice';
import { ChangeEvent, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamCpuNoiseToggle = () => {
  const dispatch = useAppDispatch();
  const shouldUseCpuNoise = useAppSelector(
    (state) => state.generation.shouldUseCpuNoise
  );
  const { t } = useTranslation();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldUseCpuNoiseChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <IAIInformationalPopover details="noiseUseCPU">
      <IAISwitch
        label={t('parameters.useCpuNoise')}
        isChecked={shouldUseCpuNoise}
        onChange={handleChange}
      />
    </IAIInformationalPopover>
  );
};

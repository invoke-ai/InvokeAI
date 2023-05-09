import { UPSCALING_LEVELS } from 'app/constants';
import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISelect from 'common/components/IAISelect';
import {
  setUpscalingLevel,
  UpscalingLevel,
} from 'features/parameters/store/postprocessingSlice';
import type { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export default function UpscaleScale() {
  const isESRGANAvailable = useAppSelector(
    (state: RootState) => state.system.isESRGANAvailable
  );

  const upscalingLevel = useAppSelector(
    (state: RootState) => state.postprocessing.upscalingLevel
  );

  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const handleChangeLevel = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setUpscalingLevel(Number(e.target.value) as UpscalingLevel));

  return (
    <IAISelect
      isDisabled={!isESRGANAvailable}
      label={t('parameters.scale')}
      value={upscalingLevel}
      onChange={handleChangeLevel}
      validValues={UPSCALING_LEVELS}
    />
  );
}

import { UPSCALING_LEVELS } from 'app/constants';
import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import {
  UpscalingLevel,
  setUpscalingLevel,
} from 'features/parameters/store/postprocessingSlice';
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

  const handleChangeLevel = (v: string) =>
    dispatch(setUpscalingLevel(Number(v) as UpscalingLevel));

  return (
    <IAIMantineSearchableSelect
      disabled={!isESRGANAvailable}
      label={t('parameters.scale')}
      value={String(upscalingLevel)}
      onChange={handleChangeLevel}
      data={UPSCALING_LEVELS}
    />
  );
}

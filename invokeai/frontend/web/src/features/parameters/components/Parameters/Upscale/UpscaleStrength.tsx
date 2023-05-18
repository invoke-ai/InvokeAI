import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setUpscalingStrength } from 'features/parameters/store/postprocessingSlice';
import { useTranslation } from 'react-i18next';

export default function UpscaleStrength() {
  const isESRGANAvailable = useAppSelector(
    (state: RootState) => state.system.isESRGANAvailable
  );
  const upscalingStrength = useAppSelector(
    (state: RootState) => state.postprocessing.upscalingStrength
  );

  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  return (
    <IAISlider
      label={`${t('parameters.upscale')} ${t('parameters.strength')}`}
      value={upscalingStrength}
      min={0}
      max={1}
      step={0.05}
      onChange={(v) => dispatch(setUpscalingStrength(v))}
      handleReset={() => dispatch(setUpscalingStrength(0.75))}
      withSliderMarks
      withInput
      withReset
      isDisabled={!isESRGANAvailable}
    />
  );
}

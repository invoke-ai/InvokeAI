import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setUpscalingDenoising } from 'features/parameters/store/postprocessingSlice';
import { useTranslation } from 'react-i18next';

export default function UpscaleDenoisingStrength() {
  const isESRGANAvailable = useAppSelector(
    (state: RootState) => state.system.isESRGANAvailable
  );

  const upscalingDenoising = useAppSelector(
    (state: RootState) => state.postprocessing.upscalingDenoising
  );

  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  return (
    <IAISlider
      label={t('parameters.denoisingStrength')}
      value={upscalingDenoising}
      min={0}
      max={1}
      step={0.01}
      onChange={(v) => {
        dispatch(setUpscalingDenoising(v));
      }}
      handleReset={() => dispatch(setUpscalingDenoising(0.75))}
      withSliderMarks
      withInput
      withReset
      isDisabled={!isESRGANAvailable}
    />
  );
}

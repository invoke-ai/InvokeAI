import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import IAISlider from 'common/components/IAISlider';
import { setCfgScale } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function MainCFGScale() {
  const dispatch = useAppDispatch();
  const cfgScale = useAppSelector(
    (state: RootState) => state.generation.cfgScale
  );
  const shouldUseSliders = useAppSelector(
    (state: RootState) => state.ui.shouldUseSliders
  );
  const { t } = useTranslation();

  const handleChangeCfgScale = (v: number) => dispatch(setCfgScale(v));

  return shouldUseSliders ? (
    <IAISlider
      label={t('parameters.cfgScale')}
      step={0.5}
      min={1.01}
      max={30}
      onChange={handleChangeCfgScale}
      handleReset={() => dispatch(setCfgScale(7.5))}
      value={cfgScale}
      sliderMarkRightOffset={-5}
      sliderNumberInputProps={{ max: 200 }}
      withInput
      withReset
      withSliderMarks
    />
  ) : (
    <IAINumberInput
      label={t('parameters.cfgScale')}
      step={0.5}
      min={1.01}
      max={200}
      onChange={handleChangeCfgScale}
      value={cfgScale}
      width="auto"
      styleClass="main-settings-block"
      textAlign="center"
      isInteger={false}
    />
  );
}

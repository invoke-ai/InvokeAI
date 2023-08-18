import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setSeamStrength } from 'features/parameters/store/generationSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSeamStrength = () => {
  const dispatch = useAppDispatch();
  const seamStrength = useAppSelector(
    (state: RootState) => state.generation.seamStrength
  );
  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters.seamStrength')}
      min={0}
      max={1}
      step={0.01}
      sliderNumberInputProps={{ max: 999 }}
      value={seamStrength}
      onChange={(v) => {
        dispatch(setSeamStrength(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => {
        dispatch(setSeamStrength(0.7));
      }}
    />
  );
};

export default memo(ParamSeamStrength);

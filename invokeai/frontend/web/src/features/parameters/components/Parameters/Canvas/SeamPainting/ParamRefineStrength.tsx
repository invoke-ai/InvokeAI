import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setRefineStrength } from 'features/parameters/store/generationSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamRefineStrength = () => {
  const dispatch = useAppDispatch();
  const refineStrength = useAppSelector(
    (state: RootState) => state.generation.refineStrength
  );
  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters.refineStrength')}
      min={0}
      max={1}
      step={0.01}
      sliderNumberInputProps={{ max: 999 }}
      value={refineStrength}
      onChange={(v) => {
        dispatch(setRefineStrength(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => {
        dispatch(setRefineStrength(0.3));
      }}
    />
  );
};

export default memo(ParamRefineStrength);

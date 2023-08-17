import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setSeamSteps } from 'features/parameters/store/generationSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSeamSteps = () => {
  const dispatch = useAppDispatch();
  const seamSteps = useAppSelector(
    (state: RootState) => state.generation.seamSteps
  );
  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters.seamSteps')}
      min={0}
      max={100}
      step={1}
      sliderNumberInputProps={{ max: 999 }}
      value={seamSteps}
      onChange={(v) => {
        dispatch(setSeamSteps(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => {
        dispatch(setSeamSteps(20));
      }}
    />
  );
};

export default memo(ParamSeamSteps);

import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setRefineSteps } from 'features/parameters/store/generationSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamRefineSteps = () => {
  const dispatch = useAppDispatch();
  const refineSteps = useAppSelector(
    (state: RootState) => state.generation.refineSteps
  );
  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters.refineSteps')}
      min={0}
      max={100}
      step={1}
      sliderNumberInputProps={{ max: 999 }}
      value={refineSteps}
      onChange={(v) => {
        dispatch(setRefineSteps(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => {
        dispatch(setRefineSteps(20));
      }}
    />
  );
};

export default memo(ParamRefineSteps);

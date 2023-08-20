import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setSeamBlur } from 'features/parameters/store/generationSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSeamBlur = () => {
  const dispatch = useAppDispatch();
  const seamBlur = useAppSelector(
    (state: RootState) => state.generation.seamBlur
  );
  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters.seamBlur')}
      min={0}
      max={64}
      step={8}
      sliderNumberInputProps={{ max: 512 }}
      value={seamBlur}
      onChange={(v) => {
        dispatch(setSeamBlur(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => {
        dispatch(setSeamBlur(8));
      }}
    />
  );
};

export default memo(ParamSeamBlur);

import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setSeamSize } from 'features/parameters/store/generationSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSeamSize = () => {
  const dispatch = useAppDispatch();
  const seamSize = useAppSelector(
    (state: RootState) => state.generation.seamSize
  );
  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters.seamSize')}
      min={0}
      max={128}
      step={8}
      sliderNumberInputProps={{ max: 512 }}
      value={seamSize}
      onChange={(v) => {
        dispatch(setSeamSize(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => {
        dispatch(setSeamSize(16));
      }}
    />
  );
};

export default memo(ParamSeamSize);

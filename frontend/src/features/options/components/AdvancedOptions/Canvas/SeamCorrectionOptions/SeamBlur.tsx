import type { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setSeamBlur } from 'features/options/store/optionsSlice';
import React from 'react';
import { useTranslation } from 'react-i18next';

export default function SeamBlur() {
  const dispatch = useAppDispatch();
  const seamBlur = useAppSelector((state: RootState) => state.options.seamBlur);
  const { t } = useTranslation();

  return (
    <IAISlider
      sliderMarkRightOffset={-4}
      label={t('options:seamBlur')}
      min={0}
      max={64}
      sliderNumberInputProps={{ max: 512 }}
      value={seamBlur}
      onChange={(v) => {
        dispatch(setSeamBlur(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => {
        dispatch(setSeamBlur(16));
      }}
    />
  );
}

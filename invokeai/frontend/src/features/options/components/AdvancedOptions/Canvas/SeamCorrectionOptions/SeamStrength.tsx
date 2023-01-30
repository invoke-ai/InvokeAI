import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setSeamStrength } from 'features/options/store/optionsSlice';
import React from 'react';
import { useTranslation } from 'react-i18next';

export default function SeamStrength() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const seamStrength = useAppSelector(
    (state: RootState) => state.options.seamStrength
  );

  return (
    <IAISlider
      sliderMarkRightOffset={-7}
      label={t('options:seamStrength')}
      min={0.01}
      max={0.99}
      step={0.01}
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
}

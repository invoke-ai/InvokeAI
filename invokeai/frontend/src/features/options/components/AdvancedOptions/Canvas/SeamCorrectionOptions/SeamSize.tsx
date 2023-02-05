import type { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setSeamSize } from 'features/options/store/optionsSlice';
import React from 'react';
import { useTranslation } from 'react-i18next';

export default function SeamSize() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const seamSize = useAppSelector((state: RootState) => state.options.seamSize);

  return (
    <IAISlider
      sliderMarkRightOffset={-6}
      label={t('options:seamSize')}
      min={1}
      max={256}
      sliderNumberInputProps={{ max: 512 }}
      value={seamSize}
      onChange={(v) => {
        dispatch(setSeamSize(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => dispatch(setSeamSize(96))}
    />
  );
}

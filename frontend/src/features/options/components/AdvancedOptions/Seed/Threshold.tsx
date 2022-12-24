import React from 'react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import { setThreshold } from 'features/options/store/optionsSlice';
import { useTranslation } from 'react-i18next';

export default function Threshold() {
  const dispatch = useAppDispatch();
  const threshold = useAppSelector(
    (state: RootState) => state.options.threshold
  );
  const { t } = useTranslation();

  const handleChangeThreshold = (v: number) => dispatch(setThreshold(v));

  return (
    <IAINumberInput
      label={t('options:noiseThreshold')}
      min={0}
      max={1000}
      step={0.1}
      onChange={handleChangeThreshold}
      value={threshold}
      isInteger={false}
    />
  );
}

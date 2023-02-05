import React from 'react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import { setPerlin } from 'features/options/store/optionsSlice';
import { useTranslation } from 'react-i18next';

export default function Perlin() {
  const dispatch = useAppDispatch();
  const perlin = useAppSelector((state: RootState) => state.options.perlin);
  const { t } = useTranslation();

  const handleChangePerlin = (v: number) => dispatch(setPerlin(v));

  return (
    <IAINumberInput
      label={t('options:perlinNoise')}
      min={0}
      max={1}
      step={0.05}
      onChange={handleChangePerlin}
      value={perlin}
      isInteger={false}
    />
  );
}

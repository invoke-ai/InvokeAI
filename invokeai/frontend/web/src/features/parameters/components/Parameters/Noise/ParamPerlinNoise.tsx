import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { setPerlin } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { perlin } = state.generation;
    return {
      perlin,
    };
  },
  defaultSelectorOptions
);

export default function ParamPerlinNoise() {
  const dispatch = useAppDispatch();
  const { perlin } = useAppSelector(selector);
  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters.perlinNoise')}
      min={0}
      max={1}
      step={0.05}
      onChange={(v) => dispatch(setPerlin(v))}
      handleReset={() => dispatch(setPerlin(0))}
      value={perlin}
      withInput
      withReset
      withSliderMarks
    />
  );
}

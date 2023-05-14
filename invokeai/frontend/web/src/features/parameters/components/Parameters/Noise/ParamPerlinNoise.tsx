import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setPerlin } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function ParamPerlinNoise() {
  const dispatch = useAppDispatch();
  const perlin = useAppSelector((state: RootState) => state.generation.perlin);
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

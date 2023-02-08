import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import { setPerlin } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function Perlin() {
  const dispatch = useAppDispatch();
  const perlin = useAppSelector((state: RootState) => state.generation.perlin);
  const { t } = useTranslation();

  const handleChangePerlin = (v: number) => dispatch(setPerlin(v));

  return (
    <IAINumberInput
      label={t('parameters:perlinNoise')}
      min={0}
      max={1}
      step={0.05}
      onChange={handleChangePerlin}
      value={perlin}
      isInteger={false}
    />
  );
}

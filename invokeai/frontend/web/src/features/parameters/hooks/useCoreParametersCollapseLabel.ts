import { useAppSelector } from 'app/store/storeHooks';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const useCoreParametersCollapseLabel = () => {
  const { t } = useTranslation();
  const shouldRandomizeSeed = useAppSelector((s) => s.generation.shouldRandomizeSeed);
  const iterations = useAppSelector((s) => s.generation.iterations);

  const iterationsLabel = useMemo(() => {
    if (iterations === 1) {
      return t('parameters.iterationsWithCount_one', { count: 1 });
    } else {
      return t('parameters.iterationsWithCount_other', { count: iterations });
    }
  }, [iterations, t]);

  const seedLabel = useMemo(() => {
    if (shouldRandomizeSeed) {
      return t('parameters.randomSeed');
    } else {
      return t('parameters.manualSeed');
    }
  }, [shouldRandomizeSeed, t]);

  const iterationsAndSeedLabel = useMemo(() => [iterationsLabel, seedLabel].join(', '), [iterationsLabel, seedLabel]);

  return { iterationsAndSeedLabel, iterationsLabel, seedLabel };
};

import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvNumberInput } from 'common/components/InvNumberInput/InvNumberInput';
import { setSeed } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamSeedNumberInput = memo(() => {
  const seed = useAppSelector((s) => s.generation.seed);
  const shouldRandomizeSeed = useAppSelector(
    (s) => s.generation.shouldRandomizeSeed
  );

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const handleChangeSeed = useCallback(
    (v: number) => dispatch(setSeed(v)),
    [dispatch]
  );

  return (
    <InvControl label={t('parameters.seed')} flexGrow={1} feature="paramSeed">
      <InvNumberInput
        step={1}
        min={NUMPY_RAND_MIN}
        max={NUMPY_RAND_MAX}
        isDisabled={shouldRandomizeSeed}
        onChange={handleChangeSeed}
        value={seed}
        flexGrow={1}
        defaultValue={0}
      />
    </InvControl>
  );
});

ParamSeedNumberInput.displayName = 'ParamSeedNumberInput';

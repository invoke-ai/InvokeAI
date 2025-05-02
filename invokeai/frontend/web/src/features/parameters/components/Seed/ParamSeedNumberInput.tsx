import { CompositeNumberInput, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectSeed, selectShouldRandomizeSeed, setSeed } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamSeedNumberInput = memo(() => {
  const seed = useAppSelector(selectSeed);
  const shouldRandomizeSeed = useAppSelector(selectShouldRandomizeSeed);

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const handleChangeSeed = useCallback((v: number) => dispatch(setSeed(v)), [dispatch]);

  return (
    <FormControl flexGrow={1} isDisabled={shouldRandomizeSeed}>
      <InformationalPopover feature="paramSeed">
        <FormLabel>{t('parameters.seed')}</FormLabel>
      </InformationalPopover>
      <CompositeNumberInput
        step={1}
        min={NUMPY_RAND_MIN}
        max={NUMPY_RAND_MAX}
        onChange={handleChangeSeed}
        value={seed}
        flexGrow={1}
        defaultValue={0}
      />
    </FormControl>
  );
});

ParamSeedNumberInput.displayName = 'ParamSeedNumberInput';

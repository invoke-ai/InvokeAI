import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSwitch } from 'common/components/InvSwitch/wrapper';
import { setShouldRandomizeSeed } from 'features/parameters/store/generationSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamSeedRandomize = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const shouldRandomizeSeed = useAppSelector(
    (s) => s.generation.shouldRandomizeSeed
  );

  const handleChangeShouldRandomizeSeed = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldRandomizeSeed(e.target.checked)),
    [dispatch]
  );

  return (
    <InvControl label={t('common.random')} w="min-content">
      <InvSwitch
        isChecked={shouldRandomizeSeed}
        onChange={handleChangeShouldRandomizeSeed}
      />
    </InvControl>
  );
});

ParamSeedRandomize.displayName = 'ParamSeedRandomize';

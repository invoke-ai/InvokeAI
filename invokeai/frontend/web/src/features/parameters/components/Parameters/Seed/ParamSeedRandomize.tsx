import { ChangeEvent, memo, useCallback } from 'react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setShouldRandomizeSeed } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';
import IAISwitch from 'common/components/IAISwitch';

const ParamSeedRandomize = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.generation.shouldRandomizeSeed
  );

  const handleChangeShouldRandomizeSeed = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldRandomizeSeed(e.target.checked)),
    [dispatch]
  );

  return (
    <IAISwitch
      label={t('common.random')}
      isChecked={shouldRandomizeSeed}
      onChange={handleChangeShouldRandomizeSeed}
    />
  );
};

export default memo(ParamSeedRandomize);

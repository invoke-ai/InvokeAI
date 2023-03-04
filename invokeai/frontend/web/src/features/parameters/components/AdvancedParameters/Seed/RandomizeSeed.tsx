import { ChangeEvent } from 'react';

import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldRandomizeSeed } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function RandomizeSeed() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.generation.shouldRandomizeSeed
  );

  const handleChangeShouldRandomizeSeed = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRandomizeSeed(e.target.checked));

  return (
    <IAISwitch
      label={t('parameters.randomizeSeed')}
      isChecked={shouldRandomizeSeed}
      onChange={handleChangeShouldRandomizeSeed}
    />
  );
}

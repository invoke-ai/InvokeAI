import { DIFFUSERS_SAMPLERS } from 'app/constants';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISelect from 'common/components/IAISelect';
import { setSampler } from 'features/parameters/store/generationSlice';
import { ChangeEvent, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const Scheduler = () => {
  const sampler = useAppSelector(
    (state: RootState) => state.generation.sampler
  );
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => dispatch(setSampler(e.target.value)),
    [dispatch]
  );

  return (
    <IAISelect
      label={t('parameters.sampler')}
      value={sampler}
      onChange={handleChange}
      validValues={DIFFUSERS_SAMPLERS}
      minWidth={36}
    />
  );
};

export default memo(Scheduler);

import { DIFFUSERS_SAMPLERS } from 'app/constants';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISelect from 'common/components/IAISelect';
import { setSampler } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export default function MainSampler() {
  const sampler = useAppSelector(
    (state: RootState) => state.generation.sampler
  );
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChangeSampler = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setSampler(e.target.value));

  return (
    <IAISelect
      label={t('parameters.sampler')}
      value={sampler}
      onChange={handleChangeSampler}
      validValues={DIFFUSERS_SAMPLERS}
      styleClass="main-settings-block"
      minWidth="9rem"
    />
  );
}

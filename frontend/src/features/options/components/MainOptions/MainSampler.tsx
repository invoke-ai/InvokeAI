import React, { ChangeEvent } from 'react';
import { SAMPLERS } from 'app/constants';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISelect from 'common/components/IAISelect';
import { setSampler } from 'features/options/store/optionsSlice';
import { useTranslation } from 'react-i18next';

export default function MainSampler() {
  const sampler = useAppSelector((state: RootState) => state.options.sampler);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChangeSampler = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setSampler(e.target.value));

  return (
    <IAISelect
      label={t('options:sampler')}
      value={sampler}
      onChange={handleChangeSampler}
      validValues={SAMPLERS}
      styleClass="main-option-block"
    />
  );
}

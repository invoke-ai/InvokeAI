import { DIFFUSERS_SAMPLERS, SAMPLERS } from 'app/constants';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISelect from 'common/components/IAISelect';
import { setSampler } from 'features/parameters/store/generationSlice';
import { activeModelSelector } from 'features/system/store/systemSelectors';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export default function MainSampler() {
  const sampler = useAppSelector(
    (state: RootState) => state.generation.sampler
  );
  const activeModel = useAppSelector(activeModelSelector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChangeSampler = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setSampler(e.target.value));

  return (
    <IAISelect
      label={t('parameters.sampler')}
      value={sampler}
      onChange={handleChangeSampler}
      validValues={
        activeModel.format === 'diffusers' ? DIFFUSERS_SAMPLERS : SAMPLERS
      }
      styleClass="main-settings-block"
      minWidth="9rem"
    />
  );
}

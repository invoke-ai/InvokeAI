import {
  DIFFUSERS_SCHEDULERS,
  IMG2IMG_DIFFUSERS_SCHEDULERS,
} from 'app/constants';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISelect from 'common/components/IAISelect';
import { setSampler } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { ChangeEvent, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSampler = () => {
  const sampler = useAppSelector(
    (state: RootState) => state.generation.sampler
  );

  const activeTabName = useAppSelector(activeTabNameSelector);

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
      validValues={
        activeTabName === 'img2img' || activeTabName == 'unifiedCanvas'
          ? IMG2IMG_DIFFUSERS_SCHEDULERS
          : DIFFUSERS_SCHEDULERS
      }
      minWidth={36}
    />
  );
};

export default memo(ParamSampler);

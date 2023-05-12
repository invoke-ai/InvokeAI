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

  const schedulers = useAppSelector((state: RootState) => state.ui.schedulers);

  const img2imgSchedulers = schedulers.filter((scheduler) => {
    return !['dpmpp_2s'].includes(scheduler);
  });

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
        ['img2img', 'unifiedCanvas'].includes(activeTabName)
          ? img2imgSchedulers
          : schedulers
      }
      minWidth={36}
    />
  );
};

export default memo(ParamSampler);

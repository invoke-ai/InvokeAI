import { Scheduler } from 'app/constants';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAICustomSelect from 'common/components/IAICustomSelect';
import { setScheduler } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamScheduler = () => {
  const scheduler = useAppSelector(
    (state: RootState) => state.generation.scheduler
  );

  const activeTabName = useAppSelector(activeTabNameSelector);

  const schedulers = useAppSelector((state: RootState) => state.ui.schedulers);

  const img2imgSchedulers = schedulers.filter((scheduler) => {
    return !['dpmpp_2s'].includes(scheduler);
  });

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: string | null | undefined) => {
      if (!v) {
        return;
      }
      dispatch(setScheduler(v as Scheduler));
    },
    [dispatch]
  );

  return (
    <IAICustomSelect
      label={t('parameters.scheduler')}
      selectedItem={scheduler}
      setSelectedItem={handleChange}
      items={
        ['img2img', 'unifiedCanvas'].includes(activeTabName)
          ? img2imgSchedulers
          : schedulers
      }
      withCheckIcon
    />
  );
};

export default memo(ParamScheduler);

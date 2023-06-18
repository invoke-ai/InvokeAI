import { SCHEDULER_NAMES, Scheduler } from 'app/constants';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { setScheduler } from 'features/parameters/store/generationSlice';
import { setSelectedSchedulers } from 'features/ui/store/uiSlice';
import { memo, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

const ParamScheduler = () => {
  const scheduler = useAppSelector(
    (state: RootState) => state.generation.scheduler
  );

  const selectedSchedulers = useAppSelector(
    (state: RootState) => state.ui.selectedSchedulers
  );

  const activeSchedulers = useAppSelector(
    (state: RootState) => state.ui.activeSchedulers
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  useEffect(() => {
    if (selectedSchedulers.length === 0) {
      dispatch(setSelectedSchedulers(SCHEDULER_NAMES));
    }

    const schedulerFound = activeSchedulers.find(
      (activeSchedulers) => activeSchedulers.value === scheduler
    );

    if (!schedulerFound) {
      dispatch(setScheduler(activeSchedulers[0].value as Scheduler));
    }
  }, [dispatch, selectedSchedulers, scheduler, activeSchedulers]);

  const handleChange = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }
      dispatch(setScheduler(v as Scheduler));
    },
    [dispatch]
  );

  return (
    <IAIMantineSelect
      label={t('parameters.scheduler')}
      value={scheduler}
      data={activeSchedulers}
      onChange={handleChange}
    />
  );
};

export default memo(ParamScheduler);

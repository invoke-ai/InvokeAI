import { createSelector } from '@reduxjs/toolkit';
import { SCHEDULER_LABEL_MAP, SCHEDULER_NAMES } from 'app/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setScheduler } from 'features/parameters/store/generationSlice';
import { SchedulerParam } from 'features/parameters/store/parameterZodSchemas';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [uiSelector, generationSelector],
  (ui, generation) => {
    const { scheduler } = generation;
    const { favoriteSchedulers: enabledSchedulers } = ui;

    const data = SCHEDULER_NAMES.map((schedulerName) => ({
      value: schedulerName,
      label: SCHEDULER_LABEL_MAP[schedulerName as SchedulerParam],
      group: enabledSchedulers.includes(schedulerName)
        ? 'Favorites'
        : undefined,
    })).sort((a, b) => a.label.localeCompare(b.label));

    return {
      scheduler,
      data,
    };
  },
  defaultSelectorOptions
);

const ParamScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { scheduler, data } = useAppSelector(selector);

  const handleChange = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }
      dispatch(setScheduler(v as SchedulerParam));
    },
    [dispatch]
  );

  return (
    <IAIMantineSelect
      label={t('parameters.scheduler')}
      value={scheduler}
      data={data}
      onChange={handleChange}
    />
  );
};

export default memo(ParamScheduler);

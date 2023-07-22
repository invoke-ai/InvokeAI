import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setScheduler } from 'features/parameters/store/generationSlice';
import {
  SCHEDULER_LABEL_MAP,
  SchedulerParam,
} from 'features/parameters/types/parameterSchemas';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { map } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [uiSelector, generationSelector],
  (ui, generation) => {
    const { scheduler } = generation;
    const { favoriteSchedulers: enabledSchedulers } = ui;

    const data = map(SCHEDULER_LABEL_MAP, (label, name) => ({
      value: name,
      label: label,
      group: enabledSchedulers.includes(name as SchedulerParam)
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
    <IAIMantineSearchableSelect
      label={t('parameters.scheduler')}
      value={scheduler}
      data={data}
      onChange={handleChange}
    />
  );
};

export default memo(ParamScheduler);

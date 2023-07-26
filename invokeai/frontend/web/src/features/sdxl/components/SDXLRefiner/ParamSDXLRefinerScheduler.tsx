import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import {
  SCHEDULER_LABEL_MAP,
  SchedulerParam,
} from 'features/parameters/types/parameterSchemas';
import { setRefinerScheduler } from 'features/sdxl/store/sdxlSlice';
import { map } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useIsRefinerAvailable } from 'services/api/hooks/useIsRefinerAvailable';

const selector = createSelector(
  stateSelector,
  ({ ui, sdxl }) => {
    const { refinerScheduler } = sdxl;
    const { favoriteSchedulers: enabledSchedulers } = ui;

    const data = map(SCHEDULER_LABEL_MAP, (label, name) => ({
      value: name,
      label: label,
      group: enabledSchedulers.includes(name as SchedulerParam)
        ? 'Favorites'
        : undefined,
    })).sort((a, b) => a.label.localeCompare(b.label));

    return {
      refinerScheduler,
      data,
    };
  },
  defaultSelectorOptions
);

const ParamSDXLRefinerScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { refinerScheduler, data } = useAppSelector(selector);
  const isRefinerAvailable = useIsRefinerAvailable();
  const handleChange = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }
      dispatch(setRefinerScheduler(v as SchedulerParam));
    },
    [dispatch]
  );

  return (
    <IAIMantineSearchableSelect
      w="100%"
      label={t('parameters.scheduler')}
      value={refinerScheduler}
      data={data}
      onChange={handleChange}
      disabled={!isRefinerAvailable}
    />
  );
};

export default memo(ParamSDXLRefinerScheduler);

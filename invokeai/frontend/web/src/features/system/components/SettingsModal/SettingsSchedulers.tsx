import { SCHEDULER_SELECT_ITEMS } from 'app/constants';
import { RootState } from 'app/store/store';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineMultiSelect from 'common/components/IAIMantineMultiSelect';
import { SchedulerParam } from 'features/parameters/store/parameterZodSchemas';
import { enabledSchedulersChanged } from 'features/ui/store/uiSlice';
import { map } from 'lodash-es';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const data = map(SCHEDULER_SELECT_ITEMS).sort((a, b) =>
  a.label.localeCompare(b.label)
);

export default function SettingsSchedulers() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const enabledSchedulers = useAppSelector(
    (state: RootState) => state.ui.enabledSchedulers
  );

  const handleChange = useCallback(
    (v: string[]) => {
      dispatch(enabledSchedulersChanged(v as SchedulerParam[]));
    },
    [dispatch]
  );

  return (
    <IAIMantineMultiSelect
      label={t('settings.availableSchedulers')}
      value={enabledSchedulers}
      data={data}
      onChange={handleChange}
      clearable
      searchable
      maxSelectedValues={99}
    />
  );
}

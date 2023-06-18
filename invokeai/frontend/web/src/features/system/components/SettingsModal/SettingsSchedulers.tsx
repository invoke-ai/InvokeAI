import { SCHEDULERS } from 'app/constants';
import { RootState } from 'app/store/store';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineMultiSelect from 'common/components/IAIMantineMultiSelect';
import { setSelectedSchedulers } from 'features/ui/store/uiSlice';
import { useTranslation } from 'react-i18next';

export default function SettingsSchedulers() {
  const dispatch = useAppDispatch();

  const selectedSchedulers = useAppSelector(
    (state: RootState) => state.ui.selectedSchedulers
  );

  const { t } = useTranslation();

  const schedulerSettingsHandler = (v: string[]) => {
    dispatch(setSelectedSchedulers(v));
  };

  return (
    <IAIMantineMultiSelect
      label={t('settings.availableSchedulers')}
      value={selectedSchedulers}
      data={SCHEDULERS}
      onChange={schedulerSettingsHandler}
      clearable
      searchable
      maxSelectedValues={99}
    />
  );
}

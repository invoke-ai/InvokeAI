import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectShowSendingToAlerts, showSendingToAlertsChanged } from 'features/system/store/systemSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const SettingsShowSendingToDifferentViewAlerts = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const isChecked = useAppSelector(selectShowSendingToAlerts);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(showSendingToAlertsChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>{t('system.showSendingToAlerts')}</FormLabel>
      <Switch isChecked={isChecked} onChange={onChange} />
    </FormControl>
  );
});

SettingsShowSendingToDifferentViewAlerts.displayName = 'SettingsShowSendingToDifferentViewAlerts';

import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectShowSendToAlerts, showSendToAlertsChanged } from 'features/system/store/systemSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const SettingsShowSendToAlerts = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const showSendToAlerts = useAppSelector(selectShowSendToAlerts);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(showSendToAlertsChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>{t('settings.showAlertsIfLost')}</FormLabel>
      <Switch isChecked={showSendToAlerts} onChange={onChange} />
    </FormControl>
  );
});

SettingsShowSendToAlerts.displayName = 'SettingsShowSendToAlerts';

import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectShowSendToToasts, showSendToToastsChanged } from 'features/system/store/systemSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const SettingsShowSendToToasts = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const isChecked = useAppSelector(selectShowSendToToasts);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(showSendToToastsChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>{t('settings.showAlertsIfLost')}</FormLabel>
      <Switch isChecked={isChecked} onChange={onChange} />
    </FormControl>
  );
});

SettingsShowSendToToasts.displayName = 'SettingsShowSendToToasts';

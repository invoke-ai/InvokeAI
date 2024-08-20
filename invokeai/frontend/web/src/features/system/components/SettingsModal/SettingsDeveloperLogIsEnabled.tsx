import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { logIsEnabledChanged } from 'features/system/store/systemSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useDispatch } from 'react-redux';

export const SettingsDeveloperLogIsEnabled = memo(() => {
  const { t } = useTranslation();
  const dispatch = useDispatch();
  const logIsEnabled = useAppSelector((s) => s.system.logIsEnabled);

  const onChangeLogIsEnabled = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(logIsEnabledChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>{t('system.enableLogging')}</FormLabel>
      <Switch isChecked={logIsEnabled} onChange={onChangeLogIsEnabled} />
    </FormControl>
  );
});

SettingsDeveloperLogIsEnabled.displayName = 'SettingsDeveloperLogIsEnabled';

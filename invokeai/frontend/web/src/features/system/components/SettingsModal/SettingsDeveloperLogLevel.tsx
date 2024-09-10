import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { isLogLevel, zLogLevel } from 'app/logging/logger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { logLevelChanged, selectSystemLogLevel } from 'features/system/store/systemSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const SettingsDeveloperLogLevel = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const logLevel = useAppSelector(selectSystemLogLevel);
  const options = useMemo(() => zLogLevel.options.map((o) => ({ label: t(`system.logLevel.${o}`), value: o })), [t]);

  const value = useMemo(() => options.find((o) => o.value === logLevel), [logLevel, options]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isLogLevel(v?.value)) {
        return;
      }
      dispatch(logLevelChanged(v.value));
    },
    [dispatch]
  );
  return (
    <FormControl>
      <FormLabel>{t('system.logLevel.logLevel')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} isSearchable={false} />
    </FormControl>
  );
});

SettingsDeveloperLogLevel.displayName = 'SettingsDeveloperLogLevel';

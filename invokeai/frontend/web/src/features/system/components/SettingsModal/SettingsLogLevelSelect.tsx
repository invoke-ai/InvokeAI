import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { isLogLevel, zLogLevel } from 'app/logging/logger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { consoleLogLevelChanged } from 'features/system/store/systemSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const SettingsLogLevelSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const consoleLogLevel = useAppSelector((s) => s.system.consoleLogLevel);
  const shouldLogToConsole = useAppSelector((s) => s.system.shouldLogToConsole);
  const options = useMemo(() => zLogLevel.options.map((o) => ({ label: o, value: o })), []);

  const value = useMemo(() => options.find((o) => o.value === consoleLogLevel), [consoleLogLevel, options]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isLogLevel(v?.value)) {
        return;
      }
      dispatch(consoleLogLevelChanged(v.value));
    },
    [dispatch]
  );
  return (
    <FormControl isDisabled={!shouldLogToConsole}>
      <FormLabel>{t('common.languagePickerLabel')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
});

SettingsLogLevelSelect.displayName = 'SettingsLogLevelSelect';

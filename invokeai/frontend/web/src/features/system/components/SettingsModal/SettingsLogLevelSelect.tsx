import { isLogLevel, zLogLevel } from 'app/logging/logger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type { InvSelectOnChange } from 'common/components/InvSelect/types';
import { consoleLogLevelChanged } from 'features/system/store/systemSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const SettingsLogLevelSelect = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const consoleLogLevel = useAppSelector(
    (state) => state.system.consoleLogLevel
  );
  const shouldLogToConsole = useAppSelector(
    (state) => state.system.shouldLogToConsole
  );
  const options = useMemo(
    () => zLogLevel.options.map((o) => ({ label: o, value: o })),
    []
  );

  const value = useMemo(
    () => options.find((o) => o.value === consoleLogLevel),
    [consoleLogLevel, options]
  );

  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!isLogLevel(v?.value)) {
        return;
      }
      dispatch(consoleLogLevelChanged(v.value));
    },
    [dispatch]
  );
  return (
    <InvControl
      label={t('common.languagePickerLabel')}
      isDisabled={!shouldLogToConsole}
    >
      <InvSelect value={value} options={options} onChange={onChange} />
    </InvControl>
  );
};

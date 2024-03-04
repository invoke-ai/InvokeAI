import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { SCHEDULER_OPTIONS } from 'features/parameters/types/constants';
import { isParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { DefaultSettingsFormData } from './DefaultSettingsForm';

type DefaultSchedulerType = DefaultSettingsFormData['scheduler'];

export function DefaultScheduler(props: UseControllerProps<DefaultSettingsFormData>) {
  const { t } = useTranslation();
  const { field } = useController(props);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterScheduler(v?.value)) {
        return;
      }
      const updatedValue = {
        ...(field.value as DefaultSchedulerType),
        value: v.value,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(
    () => SCHEDULER_OPTIONS.find((o) => o.value === (field.value as DefaultSchedulerType).value),
    [field]
  );

  const isDisabled = useMemo(() => {
    return !(field.value as DefaultSchedulerType).isEnabled;
  }, [field.value]);

  return (
    <FormControl flexDir="column" gap={1} alignItems="flex-start">
      <InformationalPopover feature="paramScheduler">
        <FormLabel>{t('parameters.scheduler')}</FormLabel>
      </InformationalPopover>
      <Combobox isDisabled={isDisabled} value={value} options={SCHEDULER_OPTIONS} onChange={onChange} />
    </FormControl>
  );
}

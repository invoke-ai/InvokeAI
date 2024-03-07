import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/DefaultSettings/SettingToggle';
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
    <FormControl flexDir="column" gap={2} alignItems="flex-start">
      <Flex justifyContent="space-between" w="full">
        <InformationalPopover feature="paramScheduler">
          <FormLabel>{t('parameters.scheduler')}</FormLabel>
        </InformationalPopover>
        <SettingToggle control={props.control} name="scheduler" />
      </Flex>
      <Combobox isDisabled={isDisabled} value={value} options={SCHEDULER_OPTIONS} onChange={onChange} />
    </FormControl>
  );
}

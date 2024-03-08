import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { isParameterPrecision } from 'features/parameters/types/parameterSchemas';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { MainModelDefaultSettingsFormData } from './MainModelDefaultSettings';

const options = [
  { label: 'FP16', value: 'fp16' },
  { label: 'FP32', value: 'fp32' },
];

type DefaultVaePrecisionType = MainModelDefaultSettingsFormData['vaePrecision'];

export function DefaultVaePrecision(props: UseControllerProps<MainModelDefaultSettingsFormData>) {
  const { t } = useTranslation();
  const { field } = useController(props);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterPrecision(v?.value)) {
        return;
      }
      const updatedValue = {
        ...(field.value as DefaultVaePrecisionType),
        value: v.value,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => options.find((o) => o.value === (field.value as DefaultVaePrecisionType).value), [field]);

  const isDisabled = useMemo(() => {
    return !(field.value as DefaultVaePrecisionType).isEnabled;
  }, [field.value]);

  return (
    <FormControl flexDir="column" gap={1} alignItems="flex-start">
      <Flex justifyContent="space-between" w="full">
        <InformationalPopover feature="paramVAEPrecision">
          <FormLabel>{t('modelManager.vaePrecision')}</FormLabel>
        </InformationalPopover>
        <SettingToggle control={props.control} name="vaePrecision" />
      </Flex>
      <Combobox isDisabled={isDisabled} value={value} options={options} onChange={onChange} />
    </FormControl>
  );
}

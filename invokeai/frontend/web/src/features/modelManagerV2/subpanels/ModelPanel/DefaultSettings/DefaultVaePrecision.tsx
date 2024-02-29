import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { isParameterPrecision, isParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { DefaultSettingsFormData } from '../DefaultSettings';
import { UseControllerProps, useController } from 'react-hook-form';

const options = [
  { label: 'FP16', value: 'fp16' },
  { label: 'FP32', value: 'fp32' },
];

type DefaultVaePrecisionType = DefaultSettingsFormData['vaePrecision'];

export function DefaultVaePrecision(props: UseControllerProps<DefaultSettingsFormData>) {
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
      <InformationalPopover feature="paramVAEPrecision">
        <FormLabel>{t('modelManager.vaePrecision')}</FormLabel>
      </InformationalPopover>
      <Combobox isDisabled={isDisabled} value={value} options={options} onChange={onChange} />
    </FormControl>
  );
}

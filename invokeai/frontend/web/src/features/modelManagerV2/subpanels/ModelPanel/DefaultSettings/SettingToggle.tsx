import { UseControllerProps, useController, useFormContext, useWatch } from 'react-hook-form';
import { DefaultSettingsFormData, FormField } from '../DefaultSettings';
import { useCallback } from 'react';
import { Switch } from '@invoke-ai/ui-library';
import { ChangeEvent } from 'react';

interface Props<T> extends UseControllerProps<DefaultSettingsFormData> {
  name: keyof DefaultSettingsFormData;
}

export function SettingToggle<T>(props: Props<T>) {
  const { field } = useController(props);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const updatedValue: FormField<T> = {
        ...(field.value as FormField<T>),
        isEnabled: e.target.checked,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  return <Switch onChange={onChange} />;
}

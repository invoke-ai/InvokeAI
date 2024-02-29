import { Switch } from '@invoke-ai/ui-library';
import type { ChangeEvent } from 'react';
import { useCallback , useMemo } from 'react';
import type { UseControllerProps} from 'react-hook-form';
import { useController } from 'react-hook-form';

import type { DefaultSettingsFormData, FormField } from './DefaultSettingsForm';

interface Props<T> extends UseControllerProps<DefaultSettingsFormData> {
  name: keyof DefaultSettingsFormData;
}

export function SettingToggle<T>(props: Props<T>) {
  const { field } = useController(props);

  const value = useMemo(() => {
    return !!(field.value as FormField<T>).isEnabled;
  }, [field.value]);

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

  return <Switch isChecked={value} onChange={onChange} />;
}

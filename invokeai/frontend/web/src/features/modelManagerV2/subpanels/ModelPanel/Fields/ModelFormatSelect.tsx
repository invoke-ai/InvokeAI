import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { LORA_MODEL_FORMAT_MAP } from 'features/parameters/types/constants';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController, useWatch } from 'react-hook-form';
import type { AnyModelConfig } from 'services/api/types';

const ModelFormatSelect = <T extends AnyModelConfig>(props: UseControllerProps<T>) => {
  const { field, formState } = useController(props);
  const type = useWatch({ control: props.control, name: 'type' });

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      field.onChange(v?.value);
    },
    [field]
  );

  const options: ComboboxOption[] = useMemo(() => {
    const modelType = type || formState.defaultValues?.type;
    if (modelType === 'lora') {
      return Object.keys(LORA_MODEL_FORMAT_MAP).map((format) => ({
        value: format,
        label: LORA_MODEL_FORMAT_MAP[format],
      })) as ComboboxOption[];
    } else if (modelType === 'embedding') {
      return [
        { value: 'embedding_file', label: 'Embedding File' },
        { value: 'embedding_folder', label: 'Embedding Folder' },
      ];
    } else if (modelType === 'ip_adapter') {
      return [{ value: 'invokeai', label: 'invokeai' }];
    } else {
      return [
        { value: 'diffusers', label: 'Diffusers' },
        { value: 'checkpoint', label: 'Checkpoint' },
      ];
    }
  }, [type, formState.defaultValues?.type]);

  const value = useMemo(() => options.find((o) => o.value === field.value), [options, field.value]);

  return <Combobox value={value} options={options} onChange={onChange} />;
};

export default typedMemo(ModelFormatSelect);

import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { MODEL_TYPE_LABELS } from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelTypeFilter';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import type { AnyModelConfig } from 'services/api/types';

const options: ComboboxOption[] = [
  { value: 'main', label: MODEL_TYPE_LABELS['main'] as string },
  { value: 'lora', label: MODEL_TYPE_LABELS['lora'] as string },
  { value: 'embedding', label: MODEL_TYPE_LABELS['embedding'] as string },
  { value: 'vae', label: MODEL_TYPE_LABELS['vae'] as string },
  { value: 'controlnet', label: MODEL_TYPE_LABELS['controlnet'] as string },
  { value: 'ip_adapter', label: MODEL_TYPE_LABELS['ip_adapter'] as string },
  { value: 't2i_adapater', label: MODEL_TYPE_LABELS['t2i_adapter'] as string },
] as const

const ModelTypeSelect = <T extends AnyModelConfig>(props: UseControllerProps<T>) => {
  const { field } = useController(props);
  const value = useMemo(() => options.find((o) => o.value === field.value), [field.value]);
  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      field.onChange(v?.value);
    },
    [field]
  );
  return <Combobox value={value} options={options} onChange={onChange} />;
};

export default typedMemo(ModelTypeSelect);

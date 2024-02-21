import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import type { AnyModelConfig } from 'services/api/types';

const options: ComboboxOption[] = [
  { value: 'sd-1', label: MODEL_TYPE_MAP['sd-1'] },
  { value: 'sd-2', label: MODEL_TYPE_MAP['sd-2'] },
  { value: 'sdxl', label: MODEL_TYPE_MAP['sdxl'] },
  { value: 'sdxl-refiner', label: MODEL_TYPE_MAP['sdxl-refiner'] },
];

const BaseModelSelect = <T extends AnyModelConfig>(props: UseControllerProps<T>) => {
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

export default typedMemo(BaseModelSelect);

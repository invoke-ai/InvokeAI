import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { useCallback, useMemo } from 'react';
import type { Control } from 'react-hook-form';
import { useController } from 'react-hook-form';
import type { UpdateModelArg } from 'services/api/endpoints/models';

const options: ComboboxOption[] = [
  { value: 'sd-1', label: MODEL_TYPE_MAP['sd-1'] },
  { value: 'sd-2', label: MODEL_TYPE_MAP['sd-2'] },
  { value: 'sdxl', label: MODEL_TYPE_MAP['sdxl'] },
  { value: 'sdxl-refiner', label: MODEL_TYPE_MAP['sdxl-refiner'] },
];

type Props = {
  control: Control<UpdateModelArg['body']>;
};

const BaseModelSelect = ({ control }: Props) => {
  const { field } = useController({ control, name: 'base' });
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

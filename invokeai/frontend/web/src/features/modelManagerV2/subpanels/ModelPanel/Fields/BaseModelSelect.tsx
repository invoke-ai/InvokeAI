import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { MODEL_BASE_TO_LONG_NAME } from 'features/modelManagerV2/models';
import { useCallback, useMemo } from 'react';
import type { Control } from 'react-hook-form';
import { useController } from 'react-hook-form';
import type { UpdateModelArg } from 'services/api/endpoints/models';

const options: ComboboxOption[] = [
  { value: 'sd-1', label: MODEL_BASE_TO_LONG_NAME['sd-1'] },
  { value: 'sd-2', label: MODEL_BASE_TO_LONG_NAME['sd-2'] },
  { value: 'sd-3', label: MODEL_BASE_TO_LONG_NAME['sd-3'] },
  { value: 'flux', label: MODEL_BASE_TO_LONG_NAME['flux'] },
  { value: 'sdxl', label: MODEL_BASE_TO_LONG_NAME['sdxl'] },
  { value: 'sdxl-refiner', label: MODEL_BASE_TO_LONG_NAME['sdxl-refiner'] },
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

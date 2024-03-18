import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { useCallback, useMemo } from 'react';
import type { Control } from 'react-hook-form';
import { useController } from 'react-hook-form';
import type { UpdateModelArg } from 'services/api/endpoints/models';

const options: ComboboxOption[] = [
  { value: 'normal', label: 'Normal' },
  { value: 'inpaint', label: 'Inpaint' },
  { value: 'depth', label: 'Depth' },
];

type Props = {
  control: Control<UpdateModelArg['body']>;
};

const ModelVariantSelect = ({ control }: Props) => {
  const { field } = useController({ control, name: 'variant' });
  const value = useMemo(() => options.find((o) => o.value === field.value), [field.value]);
  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      field.onChange(v?.value);
    },
    [field]
  );
  return <Combobox value={value} options={options} onChange={onChange} />;
};

export default typedMemo(ModelVariantSelect);

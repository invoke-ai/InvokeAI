import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { useCallback, useMemo } from 'react';
import type { Control } from 'react-hook-form';
import { useController } from 'react-hook-form';
import type { UpdateModelArg } from 'services/api/endpoints/models';

const options: ComboboxOption[] = [
  { value: 'none', label: '-' },
  { value: 'epsilon', label: 'epsilon' },
  { value: 'v_prediction', label: 'v_prediction' },
  { value: 'sample', label: 'sample' },
];

type Props = {
  control: Control<UpdateModelArg['body']>;
};

const PredictionTypeSelect = ({ control }: Props) => {
  const { field } = useController({ control, name: 'prediction_type' });
  const value = useMemo(() => options.find((o) => o.value === field.value), [field.value]);
  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      v?.value === 'none' ? field.onChange(undefined) : field.onChange(v?.value);
    },
    [field]
  );
  return <Combobox value={value} options={options} onChange={onChange} />;
};

export default typedMemo(PredictionTypeSelect);

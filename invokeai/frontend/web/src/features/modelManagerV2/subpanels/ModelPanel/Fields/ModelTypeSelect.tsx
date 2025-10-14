import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { MODEL_TYPE_TO_LONG_NAME } from 'features/modelManagerV2/models';
import { useCallback, useMemo } from 'react';
import type { Control } from 'react-hook-form';
import { useController } from 'react-hook-form';
import type { UpdateModelArg } from 'services/api/endpoints/models';
import { objectEntries } from 'tsafe';

const options: ComboboxOption[] = objectEntries(MODEL_TYPE_TO_LONG_NAME).map(([value, label]) => ({
  label,
  value,
}));

type Props = {
  control: Control<UpdateModelArg['body']>;
};

const ModelTypeSelect = ({ control }: Props) => {
  const { field } = useController({ control, name: 'type' });
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

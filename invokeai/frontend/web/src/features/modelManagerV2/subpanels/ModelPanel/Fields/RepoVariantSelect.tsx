import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import type { AnyModelConfig } from 'services/api/types';

const options: ComboboxOption[] = [
  { value: 'none', label: '-' },
  { value: 'fp16', label: 'fp16' },
  { value: 'fp32', label: 'fp32' },
];

const RepoVariantSelect = <T extends AnyModelConfig>(props: UseControllerProps<T>) => {
  const { field } = useController(props);
  const value = useMemo(() => options.find((o) => o.value === field.value), [field.value]);
  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      v?.value === 'none' ? field.onChange(undefined) : field.onChange(v?.value);
    },
    [field]
  );
  return <Combobox value={value} options={options} onChange={onChange} />;
};

export default typedMemo(RepoVariantSelect);

import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import type { CheckpointModelConfig, DiffusersModelConfig } from 'services/api/types';

const options: ComboboxOption[] = [
  { value: 'normal', label: 'Normal' },
  { value: 'inpaint', label: 'Inpaint' },
  { value: 'depth', label: 'Depth' },
];

const ModelVariantSelect = <T extends CheckpointModelConfig | DiffusersModelConfig>(props: UseControllerProps<T>) => {
  const { t } = useTranslation();
  const { field } = useController(props);
  const value = useMemo(() => options.find((o) => o.value === field.value), [field.value]);
  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      field.onChange(v?.value);
    },
    [field]
  );
  return (
    <FormControl>
      <Flex direction="column" width="full">
        <FormLabel>{t('modelManager.variant')}</FormLabel>
        <Combobox value={value} options={options} onChange={onChange} />
      </Flex>
    </FormControl>
  );
};

export default typedMemo(ModelVariantSelect);

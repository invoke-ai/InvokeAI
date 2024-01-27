import type { ChakraProps, ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { memo, useCallback, useMemo } from 'react';
import { useController, type UseControllerProps } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { useGetCheckpointConfigsQuery } from 'services/api/endpoints/models';
import type { CheckpointModelConfig } from 'services/api/types';

const sx: ChakraProps['sx'] = { w: 'full' };

const CheckpointConfigsSelect = (props: UseControllerProps<CheckpointModelConfig>) => {
  const { data } = useGetCheckpointConfigsQuery();
  const { t } = useTranslation();
  const options = useMemo<ComboboxOption[]>(() => (data ? data.map((i) => ({ label: i, value: i })) : []), [data]);
  const { field } = useController(props);
  const value = useMemo(() => options.find((o) => o.value === field.value), [field.value, options]);
  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      field.onChange(v?.value);
    },
    [field]
  );

  return (
    <FormControl>
      <FormLabel>{t('modelManager.configFile')}</FormLabel>
      <Combobox placeholder="Select A Config File" value={value} options={options} onChange={onChange} sx={sx} />
    </FormControl>
  );
};

export default memo(CheckpointConfigsSelect);

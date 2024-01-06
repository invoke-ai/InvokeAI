import type { ChakraProps } from '@chakra-ui/react';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { memo, useCallback, useMemo } from 'react';
import { useController, type UseControllerProps } from 'react-hook-form';
import { useGetCheckpointConfigsQuery } from 'services/api/endpoints/models';
import type { CheckpointModelConfig } from 'services/api/types';

const sx: ChakraProps['sx'] = { w: 'full' };

const CheckpointConfigsSelect = (
  props: UseControllerProps<CheckpointModelConfig>
) => {
  const { data } = useGetCheckpointConfigsQuery();
  const options = useMemo<InvSelectOption[]>(
    () => (data ? data.map((i) => ({ label: i, value: i })) : []),
    [data]
  );
  const { field } = useController(props);
  const value = useMemo(
    () => options.find((o) => o.value === field.value),
    [field.value, options]
  );
  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      field.onChange(v?.value);
    },
    [field]
  );

  return (
    <InvControl label="Config File">
      <InvSelect
        placeholder="Select A Config File"
        value={value}
        options={options}
        onChange={onChange}
        sx={sx}
      />
    </InvControl>
  );
};

export default memo(CheckpointConfigsSelect);

import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOption,
  InvSelectProps,
} from 'common/components/InvSelect/types';
import { useMemo } from 'react';
import { useGetCheckpointConfigsQuery } from 'services/api/endpoints/models';

type CheckpointConfigSelectProps = Omit<InvSelectProps, 'options'>;

export default function CheckpointConfigsSelect(
  props: CheckpointConfigSelectProps
) {
  const { data } = useGetCheckpointConfigsQuery();
  const options = useMemo<InvSelectOption[]>(
    () => (data ? data.map((i) => ({ label: i, value: i })) : []),
    [data]
  );
  return (
    <InvControl label="Config File">
      <InvSelect
        placeholder="Select A Config File"
        options={options}
        {...props}
      />
    </InvControl>
  );
}

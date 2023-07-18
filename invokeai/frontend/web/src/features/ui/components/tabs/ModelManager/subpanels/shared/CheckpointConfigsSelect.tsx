import IAIMantineSelect, {
  IAISelectProps,
} from 'common/components/IAIMantineSelect';
import { useGetCheckpointConfigsQuery } from 'services/api/endpoints/models';

type CheckpointConfigSelectProps = Omit<IAISelectProps, 'data'>;

export default function CheckpointConfigsSelect(
  props: CheckpointConfigSelectProps
) {
  const { data: availableCheckpointConfigs } = useGetCheckpointConfigsQuery();
  const { ...rest } = props;

  return (
    <IAIMantineSelect
      label="Config File"
      placeholder="Select A Config File"
      data={availableCheckpointConfigs ? availableCheckpointConfigs : []}
      {...rest}
    />
  );
}

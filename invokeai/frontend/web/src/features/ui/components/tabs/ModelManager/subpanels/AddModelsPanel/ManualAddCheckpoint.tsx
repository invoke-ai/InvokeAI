import { useForm } from '@mantine/form';
import { CheckpointModelConfig } from 'services/api/types';

export default function ManualAddCheckpoint() {
  const manualAddCheckpointForm = useForm<CheckpointModelConfig>({
    initialValues: {
      model_name: '',
      base_model: 'sd-1',
      model_type: 'main',
      path: '',
      description: '',
      model_format: 'checkpoint',
      error: undefined,
      vae: '',
      variant: 'normal',
      config: '',
    },
  });
  return <div>ManualAddCheckpoint</div>;
}

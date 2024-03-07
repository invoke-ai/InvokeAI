import { Badge } from '@invoke-ai/ui-library';
import { memo } from 'react';
import type { AnyModelConfig } from 'services/api/types';

type Props = {
  format: AnyModelConfig['format'];
};

const MODEL_FORMAT_NAME_MAP = {
  diffusers: 'diffusers',
  lycoris: 'lycoris',
  checkpoint: 'checkpoint',
  invokeai: 'internal',
  embedding_file: 'embedding',
  embedding_folder: 'embedding',
};

const ModelFormatBadge = ({ format }: Props) => {
  return (
    <Badge flexGrow={0} colorScheme="base" variant="subtle">
      {MODEL_FORMAT_NAME_MAP[format]}
    </Badge>
  );
};

export default memo(ModelFormatBadge);

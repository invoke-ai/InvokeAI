import { Badge } from '@invoke-ai/ui-library';
import { memo } from 'react';
import type { AnyModelConfig } from 'services/api/types';

type Props = {
  format: AnyModelConfig['format'];
};

const FORMAT_NAME_MAP: Record<AnyModelConfig['format'], string> = {
  diffusers: 'diffusers',
  lycoris: 'lycoris',
  checkpoint: 'checkpoint',
  invokeai: 'internal',
  embedding_file: 'embedding',
  embedding_folder: 'embedding',
};

const FORMAT_COLOR_MAP: Record<AnyModelConfig['format'], string> = {
  diffusers: 'base',
  lycoris: 'base',
  checkpoint: 'orange',
  invokeai: 'base',
  embedding_file: 'base',
  embedding_folder: 'base',
};

const ModelFormatBadge = ({ format }: Props) => {
  return (
    <Badge flexGrow={0} colorScheme={FORMAT_COLOR_MAP[format]} variant="subtle">
      {FORMAT_NAME_MAP[format]}
    </Badge>
  );
};

export default memo(ModelFormatBadge);

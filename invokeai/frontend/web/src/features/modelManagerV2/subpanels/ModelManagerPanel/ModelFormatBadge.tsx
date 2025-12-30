import { Badge } from '@invoke-ai/ui-library';
import type { ModelFormat } from 'features/nodes/types/common';
import { memo } from 'react';

type Props = {
  format: ModelFormat;
};

const FORMAT_NAME_MAP: Record<ModelFormat, string> = {
  diffusers: 'diffusers',
  lycoris: 'lycoris',
  checkpoint: 'checkpoint',
  invokeai: 'internal',
  embedding_file: 'embedding',
  embedding_folder: 'embedding',
  t5_encoder: 't5_encoder',
  qwen3_encoder: 'qwen3_encoder',
  bnb_quantized_int8b: 'bnb_quantized_int8b',
  bnb_quantized_nf4b: 'quantized',
  gguf_quantized: 'gguf',
  omi: 'omi',
  unknown: 'unknown',
  olive: 'olive',
  onnx: 'onnx',
};

const FORMAT_COLOR_MAP: Record<ModelFormat, string> = {
  diffusers: 'base',
  omi: 'base',
  lycoris: 'base',
  checkpoint: 'orange',
  invokeai: 'base',
  embedding_file: 'base',
  embedding_folder: 'base',
  t5_encoder: 'base',
  qwen3_encoder: 'base',
  bnb_quantized_int8b: 'base',
  bnb_quantized_nf4b: 'base',
  gguf_quantized: 'base',
  unknown: 'red',
  olive: 'base',
  onnx: 'base',
};

const ModelFormatBadge = ({ format }: Props) => {
  return (
    <Badge flexGrow={0} colorScheme={FORMAT_COLOR_MAP[format]} variant="subtle">
      {FORMAT_NAME_MAP[format]}
    </Badge>
  );
};

export default memo(ModelFormatBadge);

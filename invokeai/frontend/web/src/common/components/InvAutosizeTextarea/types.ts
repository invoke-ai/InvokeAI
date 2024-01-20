import type { TextareaProps as ChakraTextareaProps } from '@invoke-ai/ui';
import type { TextareaAutosizeProps } from 'react-textarea-autosize';

export type InvAutosizeTextareaProps = Omit<
  ChakraTextareaProps & TextareaAutosizeProps,
  'resize'
>;

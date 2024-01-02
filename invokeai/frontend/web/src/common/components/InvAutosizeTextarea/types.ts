import type { TextareaProps as ChakraTextareaProps } from '@chakra-ui/react';
import type { TextareaAutosizeProps } from 'react-textarea-autosize';

export type InvAutosizeTextareaProps = Omit<
  ChakraTextareaProps & TextareaAutosizeProps,
  'resize'
>;

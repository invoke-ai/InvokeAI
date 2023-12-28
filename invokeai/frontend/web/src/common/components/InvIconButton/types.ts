import type { IconButtonProps as ChakraIconButtonProps } from '@chakra-ui/react';
import type { ReactNode } from 'react';

export type InvIconButtonProps = ChakraIconButtonProps & {
  isChecked?: boolean;
  tooltip?: ReactNode;
};

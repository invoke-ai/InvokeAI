import type { ButtonProps } from '@chakra-ui/react';
import type { ReactNode } from 'react';

export type InvButtonProps = ButtonProps & {
  isChecked?: boolean;
  tooltip?: ReactNode;
};

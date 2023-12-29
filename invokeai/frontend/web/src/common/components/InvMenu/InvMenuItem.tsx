import { SpinnerIcon } from '@chakra-ui/icons';
import {
  forwardRef,
  keyframes,
  MenuItem as ChakraMenuItem,
} from '@chakra-ui/react';
import {memo} from'react'

import type { InvMenuItemProps } from './types';

const spin = keyframes`
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
`;

export const InvMenuItem = memo(
  forwardRef<InvMenuItemProps, typeof ChakraMenuItem>(
    (props: InvMenuItemProps, ref) => {
      const {
        isDestructive = false,
        isLoading = false,
        isDisabled,
        icon,
        ...rest
      } = props;
      return (
        <ChakraMenuItem
          ref={ref}
          icon={
            isLoading ? (
              <SpinnerIcon animation={`${spin} 1s linear infinite`} />
            ) : (
              icon
            )
          }
          isDisabled={isLoading || isDisabled}
          data-destructive={isDestructive}
          {...rest}
        />
      );
    }
  )
);

InvMenuItem.displayName = 'InvMenuItem';

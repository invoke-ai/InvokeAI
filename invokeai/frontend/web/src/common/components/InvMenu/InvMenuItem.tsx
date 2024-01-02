import { SpinnerIcon } from '@chakra-ui/icons';
import { forwardRef, MenuItem as ChakraMenuItem } from '@chakra-ui/react';
import { memo } from 'react';
import { spinAnimation } from 'theme/animations';

import type { InvMenuItemProps } from './types';

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
          icon={isLoading ? <SpinnerIcon animation={spinAnimation} /> : icon}
          isDisabled={isLoading || isDisabled}
          data-destructive={isDestructive}
          {...rest}
        />
      );
    }
  )
);

InvMenuItem.displayName = 'InvMenuItem';

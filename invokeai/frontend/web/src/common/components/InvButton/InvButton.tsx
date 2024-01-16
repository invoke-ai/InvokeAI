import { Button, forwardRef } from '@chakra-ui/react';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { memo } from 'react';

import type { InvButtonProps } from './types';

export const InvButton = memo(
  forwardRef<InvButtonProps, typeof Button>(
    ({ isChecked, tooltip, children, ...rest }: InvButtonProps, ref) => {
      if (tooltip) {
        return (
          <InvTooltip label={tooltip}>
            <Button
              ref={ref}
              colorScheme={isChecked ? 'invokeBlue' : 'base'}
              {...rest}
            >
              {children}
            </Button>
          </InvTooltip>
        );
      }

      return (
        <Button
          ref={ref}
          colorScheme={isChecked ? 'invokeBlue' : 'base'}
          {...rest}
        >
          {children}
        </Button>
      );
    }
  )
);

InvButton.displayName = 'InvButton';

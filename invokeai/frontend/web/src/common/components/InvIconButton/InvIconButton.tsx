import { forwardRef, IconButton } from '@chakra-ui/react';
import type { InvIconButtonProps } from 'common/components/InvIconButton/types';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { memo } from 'react';

export const InvIconButton = memo(
  forwardRef<InvIconButtonProps, typeof IconButton>(
    ({ isChecked, tooltip, ...rest }: InvIconButtonProps, ref) => {
      if (tooltip) {
        return (
          <InvTooltip label={tooltip}>
            <IconButton
              ref={ref}
              colorScheme={isChecked ? 'invokeBlue' : 'base'}
              {...rest}
            />
          </InvTooltip>
        );
      }

      return (
        <IconButton
          ref={ref}
          colorScheme={isChecked ? 'invokeBlue' : 'base'}
          {...rest}
        />
      );
    }
  )
);

InvIconButton.displayName = 'InvIconButton';

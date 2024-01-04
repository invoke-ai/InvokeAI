import { forwardRef, Tooltip as ChakraTooltip } from '@chakra-ui/react';
import { memo } from 'react';

import type { InvTooltipProps } from './types';

const modifiers: InvTooltipProps['modifiers'] = [
  {
    name: 'preventOverflow',
    options: {
      padding: 12,
    },
  },
];

export const InvTooltip = memo(
  forwardRef<InvTooltipProps, typeof ChakraTooltip>(
    (props: InvTooltipProps, ref) => {
      const { children, hasArrow = true, placement = 'top', ...rest } = props;
      return (
        <ChakraTooltip
          ref={ref}
          hasArrow={hasArrow}
          placement={placement}
          arrowSize={8}
          modifiers={modifiers}
          {...rest}
        >
          {children}
        </ChakraTooltip>
      );
    }
  )
);

InvTooltip.displayName = 'InvTooltip';

import { forwardRef, Tooltip as ChakraTooltip } from '@chakra-ui/react';

import type { InvTooltipProps } from './types';

export const InvTooltip = forwardRef<InvTooltipProps, typeof ChakraTooltip>(
  (props: InvTooltipProps, ref) => {
    const { children, hasArrow = true, placement = 'top', ...rest } = props;
    return (
      <ChakraTooltip
        ref={ref}
        hasArrow={hasArrow}
        placement={placement}
        {...rest}
      >
        {children}
      </ChakraTooltip>
    );
  }
);

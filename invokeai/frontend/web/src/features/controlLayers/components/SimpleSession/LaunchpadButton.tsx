import type { ButtonProps } from '@invoke-ai/ui-library';
import { Button, forwardRef } from '@invoke-ai/ui-library';
import { memo } from 'react';

export const LaunchpadButton = memo(
  forwardRef(({ children, ...rest }: ButtonProps, ref) => {
    return (
      <Button
        ref={ref}
        variant="outline"
        display="flex"
        position="relative"
        alignItems="center"
        justifyContent="left"
        borderWidth={1}
        borderRadius="base"
        p={4}
        pt={6}
        gap={2}
        w="full"
        h="full"
        {...rest}
      >
        {children}
      </Button>
    );
  })
);

LaunchpadButton.displayName = 'LaunchpadButton';

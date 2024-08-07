import type { FlexProps } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { memo } from 'react';

export const CanvasEntityHeader = memo(({ children, ...rest }: FlexProps) => {
  return (
    <Flex gap={3} alignItems="center" p={3} cursor="pointer" {...rest}>
      {children}
    </Flex>
  );
});

CanvasEntityHeader.displayName = 'CanvasEntityHeader';

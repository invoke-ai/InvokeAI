import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

export const CanvasEntitySettings = memo(({ children }: PropsWithChildren) => {
  return (
    <Flex flexDir="column" gap={3} px={3} pb={3}>
      {children}
    </Flex>
  );
});

CanvasEntitySettings.displayName = 'CanvasEntitySettings';

import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

export const CanvasEntitySettingsWrapper = memo(({ children }: PropsWithChildren) => {
  return (
    <Flex flexDir="column" gap={3} px={3} pb={3}>
      {children}
    </Flex>
  );
});

CanvasEntitySettingsWrapper.displayName = 'CanvasEntitySettingsWrapper';

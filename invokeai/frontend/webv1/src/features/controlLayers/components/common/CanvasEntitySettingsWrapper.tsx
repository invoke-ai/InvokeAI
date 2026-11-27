import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

export const CanvasEntitySettingsWrapper = memo(({ children }: PropsWithChildren) => {
  return (
    <Flex flexDir="column" gap={2} px={2} pb={2}>
      {children}
    </Flex>
  );
});

CanvasEntitySettingsWrapper.displayName = 'CanvasEntitySettingsWrapper';

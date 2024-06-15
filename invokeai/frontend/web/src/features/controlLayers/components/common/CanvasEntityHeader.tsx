import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type Props = PropsWithChildren<{ onToggle: () => void }>;

export const CanvasEntityHeader = memo(({ children, onToggle }: Props) => {
  return (
    <Flex gap={3} alignItems="center" p={3} cursor="pointer" onDoubleClick={onToggle}>
      {children}
    </Flex>
  );
});

CanvasEntityHeader.displayName = 'CanvasEntityHeader';

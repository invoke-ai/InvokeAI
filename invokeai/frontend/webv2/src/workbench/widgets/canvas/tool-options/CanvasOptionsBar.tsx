import type { ReactNode } from 'react';

import { HStack } from '@chakra-ui/react';
import { CanvasFloatingBar } from '@workbench/widgets/canvas/CanvasFloatingBar';

export const CanvasOptionsBar = ({ children }: { children?: ReactNode }) => (
  <CanvasFloatingBar maxW="full">
    <HStack align="center" gap="3" minW="0">
      {children}
    </HStack>
  </CanvasFloatingBar>
);

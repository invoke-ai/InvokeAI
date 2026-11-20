import type { SystemStyleObject } from '@chakra-ui/react';
import type { ReactNode } from 'react';

import { HStack } from '@chakra-ui/react';
import { CanvasFloatingBar } from '@workbench/widgets/canvas/CanvasFloatingBar';

export const CanvasOptionsBar = ({ children, css }: { children?: ReactNode; css?: SystemStyleObject }) => (
  <CanvasFloatingBar maxW="full">
    <HStack align="center" css={css} gap="1" minW="0">
      {children}
    </HStack>
  </CanvasFloatingBar>
);

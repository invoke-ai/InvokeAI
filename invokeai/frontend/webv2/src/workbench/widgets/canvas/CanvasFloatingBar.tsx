import type { BoxProps } from '@chakra-ui/react';
import type { PanelProps } from '@platform/ui';

import { Box } from '@chakra-ui/react';
import { Panel } from '@platform/ui';

/** Shared floating canvas chrome that re-enables pointer events; parents own positioning. */
export const CanvasFloatingBar = ({ children, ...rest }: PanelProps) => (
  <Panel density="sm" pointerEvents="auto" rounded="lg" shadow="lg" p="1" tone="surface" {...rest}>
    {children}
  </Panel>
);

export const CanvasFloatingBarDivider = (props: Pick<BoxProps, 'css'>) => (
  <Box alignSelf="stretch" bg="border.subtle" flexShrink="0" my="0.5" w="1px" {...props} />
);

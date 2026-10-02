import type { PanelProps } from '@platform/ui';

import { Box } from '@chakra-ui/react';
import { Panel } from '@platform/ui';

/**
 * Share floating canvas chrome and re-enable pointer events; parents own positioning and consumers compose their
 * own sections with {@link CanvasFloatingBarDivider}.
 */
export const CanvasFloatingBar = ({ children, ...rest }: PanelProps) => (
  <Panel density="sm" pointerEvents="auto" rounded="lg" shadow="lg" p="1" tone="surface" {...rest}>
    {children}
  </Panel>
);

/** A thin vertical rule separating groups of controls inside a {@link CanvasFloatingBar}. */
export const CanvasFloatingBarDivider = () => (
  <Box alignSelf="stretch" bg="border.subtle" flexShrink="0" my="0.5" w="1px" />
);

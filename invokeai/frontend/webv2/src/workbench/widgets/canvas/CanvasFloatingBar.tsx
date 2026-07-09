import type { PanelProps } from '@workbench/components/ui';

import { Box } from '@chakra-ui/react';
import { Panel } from '@workbench/components/ui';

/**
 * Shared floating-panel chrome for the canvas: the raised, rounded, shadowed
 * surface that the tool options bar, the staging bar, and (later) the text-tool
 * / layer-selection quick bars all sit on. It supplies only the *look* and
 * re-enables pointer events (its parent group is click-through); positioning
 * over the surface is the parent's job.
 *
 * Composition-friendly: consumers lay out their own sections inside — an
 * `HStack` of controls, `{@link CanvasFloatingBarDivider}` between groups — so
 * no per-consumer boolean props accrete here.
 */
export const CanvasFloatingBar = ({ children, ...rest }: PanelProps) => (
  <Panel density="sm" pointerEvents="auto" rounded="lg" shadow="lg" tone="raised" {...rest}>
    {children}
  </Panel>
);

/** A thin vertical rule separating groups of controls inside a {@link CanvasFloatingBar}. */
export const CanvasFloatingBarDivider = () => (
  <Box alignSelf="stretch" bg="border.subtle" flexShrink="0" my="0.5" w="1px" />
);

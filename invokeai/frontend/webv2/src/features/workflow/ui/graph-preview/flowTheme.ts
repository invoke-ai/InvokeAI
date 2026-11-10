import type { SystemStyleObject } from '@chakra-ui/react';
import type { WorkbenchThemeId } from '@theme/themes';

import { WORKFLOW_NODE_DIAMOND_ROTATION, WORKFLOW_NODE_SURFACE_TOKEN } from '@features/workflow/ui/nodeChrome';
import { THEMES_BY_ID } from '@theme/themes';

/**
 * Override xyflow light/dark defaults with inherited Chakra-backed variables; workbench variables also color grids
 * and inline handles.
 */
export const flowThemeCss: SystemStyleObject = {
  '--wb-edge-badge-fg': '{colors.fg}',
  '--wb-flow-grid': '{colors.fg.grid}',
  '--wb-node-surface': `{colors.${WORKFLOW_NODE_SURFACE_TOKEN}}`,
  '--xy-attribution-background-color': 'transparent',
  '--xy-background-color': '{colors.bg}',
  '--xy-connectionline-stroke': '{colors.accent.solid}',
  '--xy-edge-stroke': '{colors.border.emphasized}',
  '--xy-edge-stroke-selected': '{colors.accent.solid}',
  '--xy-minimap-background-color': '{colors.bg.muted}',
  '--xy-minimap-mask-background-color': '{colors.bg.inset/70}',
  '--xy-minimap-mask-stroke-color': '{colors.border.subtle}',
  '--xy-minimap-node-background-color': '{colors.bg.emphasized}',
  '--xy-selection-background-color': '{colors.accent.solid/10}',
  '--xy-selection-border': '1px dashed {colors.accent.solid}',
  // A port keeps its 12px dot but takes pointers over 24×24 centred on it. It inherits the handle's pointer events,
  // so a handle that cannot connect (previews) stays inert.
  '& .react-flow__handle::before': {
    content: '""',
    height: '24px',
    left: '50%',
    position: 'absolute',
    top: '50%',
    transform: 'translate(-50%, -50%)',
    width: '24px',
  },
  // A batch port's diamond is the handle rotated 45°; turn its grab area back so every port's is the same square.
  [`& .react-flow__handle[style*="${WORKFLOW_NODE_DIAMOND_ROTATION}"]::before`]: {
    transform: 'translate(-50%, -50%) rotate(-45deg)',
  },
};

/** The xyflow color mode matching a workbench theme. */
export const getFlowColorMode = (themeId: WorkbenchThemeId): 'light' | 'dark' => THEMES_BY_ID[themeId].colorScheme;

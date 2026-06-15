import type { SystemStyleObject } from '@chakra-ui/react';

import { THEMES_BY_ID } from '../../theme/themes';
import type { WorkbenchThemeId } from '../types';

/**
 * Theme bridge for xyflow surfaces. xyflow's stylesheet only knows light/dark,
 * so every flow renderer (workflow editor, graph preview) applies these
 * Chakra-token-backed `--xy-*` override vars on its wrapper — they inherit
 * into `.react-flow` and outrank the colorMode defaults. `--wb-flow-grid` is
 * ours, consumed by `<Background color>` and preview handles.
 */
export const flowThemeCss: SystemStyleObject = {
  '--wb-flow-grid': '{colors.fg.grid}',
  '--xy-attribution-background-color': 'transparent',
  '--xy-background-color': '{colors.bg.inset}',
  '--xy-connectionline-stroke': '{colors.accent.solid}',
  '--xy-edge-stroke': '{colors.border.emphasized}',
  '--xy-edge-stroke-selected': '{colors.accent.solid}',
  '--xy-minimap-background-color': '{colors.bg.muted}',
  '--xy-minimap-mask-background-color': '{colors.bg.inset/70}',
  '--xy-minimap-mask-stroke-color': '{colors.border.subtle}',
  '--xy-minimap-node-background-color': '{colors.bg.emphasized}',
  '--xy-selection-background-color': '{colors.accent.solid/10}',
  '--xy-selection-border': '1px dashed {colors.accent.solid}',
};

/** The xyflow color mode matching a workbench theme. */
export const getFlowColorMode = (themeId: WorkbenchThemeId): 'light' | 'dark' => THEMES_BY_ID[themeId].colorScheme;

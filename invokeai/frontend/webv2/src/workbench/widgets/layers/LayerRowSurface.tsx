import type { RowProps } from '@platform/ui';
import type { ReactNode } from 'react';

import { Row } from '@platform/ui';
import { useMemo } from 'react';

import { LAYER_TREE_INDENT_PX } from './layerPanelRows';

const LAYER_ROW_BACKGROUND_TRANSITION = 'background min(40ms, var(--wb-motion-duration-fast)) ease-out';
/**
 * Rows and stack headers end with this inset and space their trailing buttons by this gap, so the header eye and New
 * buttons share columns with the rows' eyes and locks. Nesting indents the start edge only.
 */
export const LAYER_PANEL_END_INSET = '1.5';
export const LAYER_PANEL_CONTROL_GAP = '0.5';
/**
 * Tree rows keep a gap below their surface (`pb="0.5"`); the ring frames the surface, not the gap, so the row's
 * content stays centred inside it.
 */
export const TREE_ROW_FOCUS = {
  _after: {
    borderRadius: 'sm',
    bottom: '0.5',
    boxShadow: 'inset 0 0 0 2px {colors.accent.solid}',
    content: '""',
    insetInline: '0',
    pointerEvents: 'none',
    position: 'absolute',
    top: '0',
  },
  outline: 'none',
};
// The leading inset mirrors the end inset.
const LAYER_ROW_INSET_PX = 6;

export const LayerRowSurface = ({
  active,
  children,
  cursor = 'default',
  depth,
}: {
  active: RowProps['active'];
  children: ReactNode;
  cursor?: 'default' | 'grabbing';
  depth: number;
}) => {
  const indentStyle = useMemo(
    () => ({ paddingLeft: `${LAYER_ROW_INSET_PX + depth * LAYER_TREE_INDENT_PX}px` }),
    [depth]
  );
  return (
    <Row
      active={active}
      alignItems="center"
      cursor={cursor}
      display="flex"
      gap="1.5"
      h="full"
      pe={LAYER_PANEL_END_INSET}
      style={indentStyle}
      transition={LAYER_ROW_BACKGROUND_TRANSITION}
    >
      {children}
    </Row>
  );
};

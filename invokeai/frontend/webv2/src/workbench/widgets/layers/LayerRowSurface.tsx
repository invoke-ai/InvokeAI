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
 * Tree rows keep a gap below their surface (`pb="0.5"`) and indent it past the tree guides; the ring frames the
 * surface, not the gap or the guides, so it matches the selection fill and the row's content stays centred inside it.
 */
export const TREE_ROW_FOCUS = {
  '& > [data-layer-row-surface]': { boxShadow: 'inset 0 0 0 2px {colors.accent.solid}' },
  outline: 'none',
};
// The leading inset mirrors the end inset.
export const LAYER_ROW_INSET_PX = 6;
/** Layer and group rows lead with a box this wide: thumbnail, group pile or degraded placeholder. */
export const LAYER_LEADING_BOX_PX = 32;

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
  // Nesting is a margin, not padding: the selection fill starts right of the tree guides.
  const indentStyle = useMemo(() => ({ marginLeft: `${depth * LAYER_TREE_INDENT_PX}px`, width: 'auto' }), [depth]);
  return (
    <Row
      active={active}
      alignItems="center"
      cursor={cursor}
      data-layer-row-surface=""
      display="flex"
      gap="1.5"
      h="full"
      pe={LAYER_PANEL_END_INSET}
      ps={`${LAYER_ROW_INSET_PX}px`}
      style={indentStyle}
      transition={LAYER_ROW_BACKGROUND_TRANSITION}
    >
      {children}
    </Row>
  );
};

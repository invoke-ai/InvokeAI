import type { RowProps } from '@platform/ui';
import type { ReactNode } from 'react';

import { Row } from '@platform/ui';
import { rowFocusRing } from '@theme/recipes';
import { useMemo } from 'react';

import { LAYER_TREE_INDENT_PX } from './layerPanelRows';

/** Shared end inset and gap, so header and row trailing buttons line up in columns. */
export const LAYER_PANEL_END_INSET = '1.5';
export const LAYER_PANEL_CONTROL_GAP = '0.5';
/** The ring frames the surface, not the gap below it or the guides, matching the selection fill. */
export const TREE_ROW_FOCUS = {
  '& > [data-layer-row-surface]': rowFocusRing,
  outline: 'none',
};
// The leading inset mirrors the end inset.
export const LAYER_ROW_INSET_PX = 3;
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
    >
      {children}
    </Row>
  );
};

import type { CSSProperties } from 'react';

import { Box } from '@chakra-ui/react';
import { memo } from 'react';

import { LAYER_TREE_INDENT_PX, TREE_GUIDE } from './layerPanelRows';
import { LAYER_LEADING_BOX_PX, LAYER_ROW_INSET_PX } from './LayerRowSurface';

const GUIDE_COLOR = 'border.subtle';
const ELBOW_RADIUS = '4px';
/** Column `c`'s 1px line sits under the centre of the leading box of rows at depth `c`. */
const columnLeft = (column: number): number =>
  column * LAYER_TREE_INDENT_PX + LAYER_ROW_INSET_PX + LAYER_LEADING_BOX_PX / 2 - 1;
/** A branch ends where the next level's surface (and its selection fill) begins. */
const BRANCH_WIDTH_PX = LAYER_TREE_INDENT_PX - columnLeft(0);
/**
 * Rows keep a 2px gap below their surface, so the leading box is centred at `50% - 1px`. The branch's 1px stroke
 * fills the pixel just below that centre, and a stem starts 2px under the parent's leading box.
 */
const BRANCH_HEIGHT = '50%';
const STEM_TOP = `calc(50% - 1px + ${LAYER_LEADING_BOX_PX / 2 + 2}px)`;

const lineStyle = (column: number): CSSProperties => ({ left: columnLeft(column) });
const elbowStyle = (column: number): CSSProperties => ({ left: columnLeft(column), width: BRANCH_WIDTH_PX });

/** A full-height line through the row, or a stem from beneath the row's own leading box to its end. */
const Line = ({ column, stem = false }: { column: number; stem?: boolean }) => (
  <Box
    bg={GUIDE_COLOR}
    bottom="0"
    data-tree-guide={stem ? 'stem' : 'line'}
    position="absolute"
    style={lineStyle(column)}
    top={stem ? STEM_TOP : '0'}
    w="1px"
  />
);

const Elbow = ({ column }: { column: number }) => (
  <Box
    borderBottomLeftRadius={ELBOW_RADIUS}
    borderBottomWidth="1px"
    borderColor={GUIDE_COLOR}
    borderLeftWidth="1px"
    data-tree-guide="branch"
    h={BRANCH_HEIGHT}
    position="absolute"
    style={elbowStyle(column)}
    top="0"
  />
);

const Guide = ({ column, glyph }: { column: number; glyph: string }) => {
  switch (glyph) {
    case TREE_GUIDE.pass:
      return <Line column={column} />;
    case TREE_GUIDE.tee:
      return (
        <>
          <Line column={column} />
          <Elbow column={column} />
        </>
      );
    case TREE_GUIDE.elbow:
      return <Elbow column={column} />;
    case TREE_GUIDE.stem:
      return <Line column={column} stem />;
    default:
      return null;
  }
};

/**
 * One row's connector lines, drawn behind it in its virtual slot. Each row draws only its own full-height segments,
 * so consecutive rows join without knowing about each other. Decorative: hidden from assistive technology and
 * transparent to the pointer, so hit-testing, drag and drop and focus see only the row.
 */
const LayerTreeGuidesComponent = ({ guides }: { guides: string }) => (
  <Box aria-hidden data-tree-guides="" inset="0" pointerEvents="none" position="absolute">
    {[...guides].map((glyph, column) => (
      // Columns are positional: a column's glyph changes in place, never moves.
      <Guide key={column} column={column} glyph={glyph} />
    ))}
  </Box>
);

export const LayerTreeGuides = memo(LayerTreeGuidesComponent);

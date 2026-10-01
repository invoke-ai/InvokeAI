import { chakra } from '@chakra-ui/react';

import { LIST_DIVIDER_INSET, LIST_ROW_GAP_PX, LIST_ROW_INSET } from './listLayout';

const HAIRLINE_PX = 1;
/** Softer than a section border: it separates rows, it does not frame them. */
const DIVIDER_COLOR = 'border.subtle/50';
/** Centres the hairline in the row gap. */
const HALF_GAP = `${(LIST_ROW_GAP_PX - HAIRLINE_PX) / 2}px`;

// A border, not a 1px fill: borders snap to whole device pixels at fractional zoom, fills can render 2px.
const HAIRLINE = {
  borderColor: DIVIDER_COLOR,
  borderTopStyle: 'solid',
  borderTopWidth: `${HAIRLINE_PX}px`,
  h: 0,
} as const;

const IN_FLOW_CSS = {
  ...HAIRLINE,
  flexShrink: 0,
  mx: LIST_DIVIDER_INSET,
  my: HALF_GAP,
} as const;

/** For virtualized rows: drawn inside the row slot's bottom gap, clear of the row's own inset. */
const IN_SLOT_CSS = {
  ...HAIRLINE,
  bottom: HALF_GAP,
  insetInline: `calc(${LIST_ROW_INSET} + ${LIST_DIVIDER_INSET})`,
  pointerEvents: 'none',
  position: 'absolute',
} as const;

/** Hairline between two rows; decorative, so hidden from assistive tech. */
export const ListDivider = ({ placement }: { placement: 'in-flow' | 'in-slot' }) => (
  <chakra.div aria-hidden css={placement === 'in-flow' ? IN_FLOW_CSS : IN_SLOT_CSS} />
);

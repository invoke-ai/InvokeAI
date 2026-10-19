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
  transition: 'opacity var(--wb-motion-duration-fast) ease',
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

// A row shows its pointed fill while hovered or while its context menu is open; the hairlines either side of it hide
// so that fill reads as one surface rather than a band between lines. `:has()` cannot nest, so a row wrapped by its
// caller (menu state on a descendant) is matched as "a sibling containing it" instead.
const POINTED = '[data-list-surface]:not([data-static]):is(:hover, [data-menu-open])';
const HIDDEN = { opacity: 0 } as const;

/** For a container of in-flow rows and dividers (`ListStack`); rows may be wrapped by their caller. */
export const IN_FLOW_DIVIDER_HIDING_CSS = {
  [[
    `& > ${POINTED} + [data-list-divider]`,
    `& > :has(${POINTED}) + [data-list-divider]`,
    `& > [data-list-divider]:has(+ ${POINTED})`,
    `& > [data-list-divider]:has(+ * ${POINTED})`,
  ].join(', ')]: HIDDEN,
} as const;

/** For a container of row slots that each draw the divider below themselves (`List`). */
export const IN_SLOT_DIVIDER_HIDING_CSS = {
  [[`& > :has(${POINTED}) > [data-list-divider]`, `& > :has(+ * ${POINTED}) > [data-list-divider]`].join(', ')]: HIDDEN,
} as const;

/** Hairline between two rows; decorative, so hidden from assistive tech. */
export const ListDivider = ({ placement }: { placement: 'in-flow' | 'in-slot' }) => (
  <chakra.div aria-hidden css={placement === 'in-flow' ? IN_FLOW_CSS : IN_SLOT_CSS} data-list-divider="" />
);

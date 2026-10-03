/**
 * Invoke sits on top of its attached neighbours with full control corners. Each neighbour hides the edge facing
 * Invoke (it would show in the corners as a bracket) and runs a tail of its frame under the button, so the corners
 * reveal a continuing border instead of a notch.
 */

/** Doubled specificity outranks the attached Group's `!important` corner reset. */
export const INVOKE_BUTTON_CSS = { '&&': { borderRadius: 'control !important' } } as const;

const TAIL = {
  borderBlockWidth: '1px',
  content: '""',
  pointerEvents: 'none',
  position: 'absolute',
} as const;

/**
 * For the number input before Invoke. Its surface is a transparent outline input (a replaced element, so the root
 * carries the tail); the tail overlaps the input by 2px so the opaque border rows join without a sub-pixel seam.
 */
export const TAIL_UNDER_INVOKE_AFTER_CSS = {
  '& input': { borderInlineEndColor: 'transparent' },
  '&::after': {
    ...TAIL,
    borderColor: 'border',
    insetBlock: 0,
    insetInlineStart: 'calc(100% - 2px)',
    width: 'calc({radii.control} + 2px)',
  },
} as const;

/**
 * For a button after Invoke; its edge stays as a transparent 1px border so the content box stays centred. The tail
 * inherits the surface so hover and active states carry under the corners. It covers that edge column with its border
 * rows (no mitred corner) but clips its fill off the column, where the button's translucent fill would double.
 */
export const TAIL_UNDER_INVOKE_BEFORE_CSS = {
  borderInlineStartColor: 'transparent',
  '&::before': {
    ...TAIL,
    backgroundClip: 'content-box',
    backgroundColor: 'inherit',
    borderBlockColor: 'inherit',
    // Offsets resolve inside the button's border; step out so the tail continues its rows and spans its edge column.
    insetBlock: '-1px',
    insetInlineEnd: '100%',
    paddingInlineEnd: '1px',
    width: 'calc({radii.control} + 2px)',
  },
} as const;

import { VisuallyHidden } from '@chakra-ui/react';
import { ToggleDot } from '@platform/ui/ToggleDot';
import { useId } from 'react';

/** The same 24px hit target as the row's small ghost icon buttons, so the trailing columns share one rhythm. */
export const LAYER_ACTIVE_DOT_SIZE = '6';

const DOT_TRANSITION = 'background var(--wb-motion-duration-fast), border-color var(--wb-motion-duration-fast)';
const DOT_BASE = {
  borderRadius: 'full',
  borderWidth: '1px',
  content: '""',
  h: '3',
  insetInlineStart: '50%',
  position: 'absolute',
  top: '50%',
  transform: 'translate(-50%, -50%)',
  transition: DOT_TRANSITION,
  w: '3',
};
const DOT_CHECKED = { ...DOT_BASE, bg: 'accent.solid', borderColor: 'accent.solid' };
/**
 * On by itself, but an ancestor keeps the node out: a dashed accent ring, so the state reads by shape as well as color
 * against the solid grey ring of a dot that is off.
 */
const DOT_GATED = {
  ...DOT_BASE,
  bg: 'transparent',
  borderColor: 'accent.solid',
  borderStyle: 'dashed',
  borderWidth: '1.5px',
};
// Off is the faintest state of an on/off control, so its ring keeps at least 3:1 against every row fill in every theme
// (LayerActiveDot.browser.test.tsx measures it); grey and hollow, it still reads apart from the filled accent of on.
const DOT_UNCHECKED = { ...DOT_BASE, bg: 'transparent', borderColor: 'fg.muted' };
// The button fills like the ghost icon buttons beside it. A dot that is on keeps its accent under the pointer: the
// darker hover accent of a plain ToggleDot sinks into that fill and reads as the dot switching off.
const HOVER_FILL = 'gray.hoverTint/10';
const HOVER_CHECKED = { bg: HOVER_FILL };
const HOVER_GATED = { bg: HOVER_FILL };
const HOVER_UNCHECKED = { _before: { borderColor: 'fg' }, bg: HOVER_FILL };
// Read-only while editing is locked: the dot dims as disabled icon buttons do and takes no hover fill.
const HOVER_NONE = {};
const dimmed = (style: typeof DOT_BASE) => ({ ...style, opacity: 0.5 });

const dotStyle = (checked: boolean, gated: boolean) => (checked ? (gated ? DOT_GATED : DOT_CHECKED) : DOT_UNCHECKED);
const hoverStyle = (checked: boolean, gated: boolean) =>
  checked ? (gated ? HOVER_GATED : HOVER_CHECKED) : HOVER_UNCHECKED;

const stopPropagation = (event: { stopPropagation: () => void }): void => event.stopPropagation();

/**
 * A layer row's "takes part in generation" toggle, reading on, gated by an ancestor, and off distinctly. Pointer-only
 * like every row control (the row's menu carries the same action), and it never starts a row drag.
 */
export const LayerActiveDot = ({
  checked,
  disabled,
  gatedReason,
  label,
  onCheckedChange,
  tooltip,
}: {
  checked: boolean;
  disabled: boolean;
  /**
   * Set when the dot is checked but an ancestor keeps the node out: why, shown as the tooltip and given to assistive
   * tech as the dot's description, since `aria-pressed` alone would report the node as taking part.
   */
  gatedReason: string | null;
  label: string;
  onCheckedChange: (checked: boolean) => void;
  /** The action a click takes, while nothing above gates the dot. */
  tooltip: string;
}) => {
  const descriptionId = useId();
  const gated = checked && gatedReason !== null;
  return (
    <>
      <ToggleDot
        _before={disabled ? dimmed(dotStyle(checked, gated)) : dotStyle(checked, gated)}
        _hover={disabled ? HOVER_NONE : hoverStyle(checked, gated)}
        bg="transparent"
        borderWidth="0"
        aria-describedby={gated ? descriptionId : undefined}
        checked={checked}
        disabled={disabled}
        h={LAYER_ACTIVE_DOT_SIZE}
        label={label}
        position="relative"
        rounded="control"
        tabIndex={-1}
        tooltip={gated ? gatedReason : tooltip}
        transition={DOT_TRANSITION}
        w={LAYER_ACTIVE_DOT_SIZE}
        onCheckedChange={onCheckedChange}
        onPointerDown={stopPropagation}
      />
      {gated ? <VisuallyHidden id={descriptionId}>{gatedReason}</VisuallyHidden> : null}
    </>
  );
};

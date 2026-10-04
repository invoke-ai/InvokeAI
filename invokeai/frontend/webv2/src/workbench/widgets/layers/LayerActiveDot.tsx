import type { MouseEvent } from 'react';

import { Box } from '@chakra-ui/react';
import { ToggleDot } from '@platform/ui';

export const ROW_SELECTION_FOCUS = { outline: '2px solid', outlineColor: 'accent.solid', outlineOffset: '-2px' };

const DOT_BASE = {
  borderRadius: 'full',
  borderWidth: '1px',
  content: '""',
  h: '3',
  inset: '50% auto auto calc(50% + 1.5px)',
  position: 'absolute',
  transform: 'translate(-50%, -50%)',
  transition: 'background var(--wb-motion-duration-fast), border-color var(--wb-motion-duration-fast)',
  w: '3',
};
const DOT_CHECKED = { ...DOT_BASE, bg: 'accent.solid', borderColor: 'accent.solid' };
/** Enabled on its own, but an ancestor keeps it out of the composite. */
const DOT_GATED = { ...DOT_BASE, bg: 'transparent', borderColor: 'accent.solid' };
const DOT_UNCHECKED = { ...DOT_BASE, bg: 'transparent', borderColor: 'border.emphasized' };
const DOT_CHECKED_HOVER = { _before: { bg: 'accent.emphasized', borderColor: 'accent.emphasized' } };
const DOT_UNCHECKED_HOVER = { _before: { borderColor: 'fg.muted' } };

const stopPropagation = (event: { stopPropagation: () => void }): void => event.stopPropagation();

/** Show contributing, ancestor-gated, and off states distinctly; keep toggle clicks/focus off the tree item. */
export const LayerActiveDot = ({
  checked,
  disabled,
  gated,
  label,
  onCheckedChange,
  onKeepRowFocus,
  tooltip,
}: {
  checked: boolean;
  disabled: boolean;
  /** Checked, but an ancestor keeps the node out of the composite. */
  gated: boolean;
  label: string;
  onCheckedChange: (checked: boolean) => void;
  onKeepRowFocus: (event: MouseEvent<HTMLElement>) => void;
  tooltip?: string;
}) => (
  <Box
    display="flex"
    flexShrink="0"
    onClick={stopPropagation}
    onMouseDown={onKeepRowFocus}
    onPointerDown={stopPropagation}
  >
    <ToggleDot
      _before={checked ? (gated ? DOT_GATED : DOT_CHECKED) : DOT_UNCHECKED}
      _focusVisible={ROW_SELECTION_FOCUS}
      _hover={checked ? DOT_CHECKED_HOVER : DOT_UNCHECKED_HOVER}
      bg="transparent"
      borderWidth="0"
      checked={checked}
      cursor={disabled ? 'not-allowed' : undefined}
      disabled={disabled}
      h="6"
      label={label}
      position="relative"
      tabIndex={-1}
      tooltip={tooltip}
      transition="none"
      w="6"
      onCheckedChange={onCheckedChange}
    />
  </Box>
);

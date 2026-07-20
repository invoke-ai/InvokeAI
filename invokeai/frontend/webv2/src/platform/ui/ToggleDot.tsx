import type { BoxProps } from '@chakra-ui/react';
import type { ReactNode } from 'react';

import { Box } from '@chakra-ui/react';
import { useCallback } from 'react';

import { Tooltip } from './Tooltip';

const TOGGLE_DOT_HOVER_STYLES = { borderColor: 'fg.muted' };
const TOGGLE_DOT_CHECKED_HOVER_STYLES = { bg: 'accent.emphasized', borderColor: 'accent.emphasized' };

export interface ToggleDotProps extends Omit<BoxProps, 'aria-label' | 'as' | 'children' | 'onChange' | 'onClick'> {
  checked: boolean;
  disabled?: boolean;
  label: string;
  tooltip?: ReactNode;
  onCheckedChange: (checked: boolean) => void;
}

/** Small pressable dot for compact enabled/disabled state toggles. */
export const ToggleDot = ({
  checked,
  disabled = false,
  label,
  onCheckedChange,
  tooltip = label,
  ...rest
}: ToggleDotProps) => {
  const handleClick = useCallback(() => onCheckedChange(!checked), [checked, onCheckedChange]);
  const nativeDisabled = disabled ? { disabled: true } : {};
  const dot = (
    <Box
      as="button"
      {...nativeDisabled}
      aria-disabled={disabled || undefined}
      aria-label={label}
      aria-pressed={checked}
      bg={checked ? 'accent.solid' : 'transparent'}
      borderColor={checked ? 'accent.solid' : 'border.emphasized'}
      borderWidth="1px"
      cursor={disabled ? 'not-allowed' : 'pointer'}
      flexShrink="0"
      h="3"
      rounded="full"
      transition="background var(--wb-motion-duration-fast), border-color var(--wb-motion-duration-fast)"
      w="3"
      _hover={checked ? TOGGLE_DOT_CHECKED_HOVER_STYLES : TOGGLE_DOT_HOVER_STYLES}
      onClick={disabled ? undefined : handleClick}
      {...rest}
    />
  );

  return tooltip ? <Tooltip content={tooltip}>{dot}</Tooltip> : dot;
};

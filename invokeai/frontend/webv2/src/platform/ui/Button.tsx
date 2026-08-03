import type { ComponentProps, ReactNode } from 'react';

import {
  Button as ChakraButton,
  CloseButton as ChakraCloseButton,
  Icon,
  IconButton as ChakraIconButton,
} from '@chakra-ui/react';
import { useCallback } from 'react';

import { Tooltip } from './Tooltip';

type ButtonProps = ComponentProps<typeof ChakraButton>;
type CloseButtonProps = ComponentProps<typeof ChakraCloseButton>;
type ChakraIconButtonProps = ComponentProps<typeof ChakraIconButton>;

type IconButtonAccessibleName =
  | {
      'aria-label': string;
      'aria-labelledby'?: never;
    }
  | {
      'aria-label'?: never;
      'aria-labelledby': string;
    };

export type IconButtonProps = Omit<ChakraIconButtonProps, 'aria-label' | 'aria-labelledby'> & IconButtonAccessibleName;

/**
 * Workbench buttons. Solid buttons default to the blue `accent` palette; every
 * other variant stays on the neutral, theme-aware `gray` palette. Pass
 * `colorPalette` explicitly to override (e.g. `brand` for the global Invoke
 * action, `red` for destructive actions).
 */
const defaultPalette = (variant: ButtonProps['variant']): ButtonProps['colorPalette'] =>
  variant === undefined || variant === 'solid' ? 'accent' : 'gray';

export const Button = ({ colorPalette, ...props }: ButtonProps) => (
  <ChakraButton colorPalette={colorPalette ?? defaultPalette(props.variant)} {...props} />
);

export const IconButton = ({ colorPalette, ...props }: IconButtonProps) => (
  <ChakraIconButton colorPalette={colorPalette ?? defaultPalette(props.variant)} {...props} />
);

export const CloseButton = (props: CloseButtonProps) => <ChakraCloseButton {...props} />;

export interface ToggleIconButtonProps extends Omit<
  IconButtonProps,
  'aria-label' | 'aria-labelledby' | 'aria-pressed' | 'children' | 'onClick'
> {
  checked: boolean;
  icon: React.ElementType;
  /** Names the control for assistive tech and, unless overridden, the tooltip. */
  label: string;
  tooltip?: ReactNode;
  onCheckedChange: (checked: boolean) => void;
}

/**
 * A compact on/off control that states what it toggles. Prefer this over
 * {@link ToggleDot} wherever the setting is not self-evident from its
 * neighbours — the icon carries the meaning, `aria-pressed` carries the state,
 * and the filled variant makes "on" readable at a glance.
 */
export const ToggleIconButton = ({
  checked,
  icon,
  label,
  onCheckedChange,
  tooltip = label,
  ...props
}: ToggleIconButtonProps) => {
  const handleClick = useCallback(() => onCheckedChange(!checked), [checked, onCheckedChange]);

  return (
    <Tooltip content={tooltip}>
      <IconButton
        aria-label={label}
        aria-pressed={checked}
        color={checked ? undefined : 'fg.muted'}
        size="2xs"
        variant={checked ? 'solid' : 'ghost'}
        {...props}
        onClick={handleClick}
      >
        <Icon as={icon} boxSize="3.5" />
      </IconButton>
    </Tooltip>
  );
};

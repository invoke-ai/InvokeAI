import type { ComponentProps } from 'react';

import {
  Button as ChakraButton,
  CloseButton as ChakraCloseButton,
  IconButton as ChakraIconButton,
} from '@chakra-ui/react';

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

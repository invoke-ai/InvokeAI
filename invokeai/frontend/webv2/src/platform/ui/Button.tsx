import type { ComponentProps, ReactNode } from 'react';

import {
  Button as ChakraButton,
  CloseButton as ChakraCloseButton,
  Icon,
  IconButton as ChakraIconButton,
  Stack,
  Text,
} from '@chakra-ui/react';
import { useCallback, useId, useMemo } from 'react';

import { Tooltip } from './Tooltip';

interface UnavailableProps {
  /**
   * Why the control cannot act now (say, no clipboard on a plain-HTTP origin). It stays visible and focusable, reads
   * as disabled (`aria-disabled`, described by the reason), ignores clicks, and shows the reason in its tooltip. Native
   * `disabled` stays for states the user can change, where there is nothing to explain.
   */
  unavailableReason?: string;
}

type ButtonProps = ComponentProps<typeof ChakraButton> & UnavailableProps;
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

export type IconButtonProps = Omit<ChakraIconButtonProps, 'aria-label' | 'aria-labelledby'> &
  IconButtonAccessibleName &
  UnavailableProps & {
    /** The platform tooltip an icon-only control carries; while unavailable it adds the reason. */
    tooltip?: ReactNode;
  };

/** Solid defaults to accent; other variants to gray. Explicit colorPalette overrides both. */
const defaultPalette = (variant: ButtonProps['variant']): ButtonProps['colorPalette'] =>
  variant === undefined || variant === 'solid' ? 'accent' : 'gray';

/** The props that make a control unavailable, and the hidden copy of the reason its description points at. */
const useUnavailable = (reason: string | undefined) => {
  const descriptionId = useId();
  return useMemo(
    () =>
      reason
        ? {
            description: (
              <span hidden id={descriptionId}>
                {reason}
              </span>
            ),
            props: { 'aria-describedby': descriptionId, 'aria-disabled': true } as const,
          }
        : null,
    [descriptionId, reason]
  );
};

export const Button = ({ children, colorPalette, onClick, unavailableReason, ...props }: ButtonProps) => {
  const unavailable = useUnavailable(unavailableReason);
  const button = (
    <ChakraButton
      colorPalette={colorPalette ?? defaultPalette(props.variant)}
      {...props}
      {...unavailable?.props}
      onClick={unavailable ? undefined : onClick}
    >
      {children}
      {unavailable?.description}
    </ChakraButton>
  );

  return unavailableReason ? <Tooltip content={unavailableReason}>{button}</Tooltip> : button;
};

export const IconButton = ({
  children,
  colorPalette,
  onClick,
  tooltip,
  unavailableReason,
  ...props
}: IconButtonProps) => {
  const unavailable = useUnavailable(unavailableReason);
  const content = useMemo(
    () =>
      unavailableReason ? (
        <Stack gap="0">
          {tooltip ? <Text>{tooltip}</Text> : null}
          <Text color={tooltip ? 'fg.muted' : undefined}>{unavailableReason}</Text>
        </Stack>
      ) : (
        tooltip
      ),
    [tooltip, unavailableReason]
  );
  const button = (
    <ChakraIconButton
      colorPalette={colorPalette ?? defaultPalette(props.variant)}
      {...props}
      {...unavailable?.props}
      onClick={unavailable ? undefined : onClick}
    >
      {children}
      {unavailable?.description}
    </ChakraIconButton>
  );

  return content ? <Tooltip content={content}>{button}</Tooltip> : button;
};

/** Dismissal chrome is muted. */
export const CloseButton = (props: CloseButtonProps) => <ChakraCloseButton color="fg.muted" {...props} />;

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

/** Use an icon toggle when nearby context cannot explain ToggleDot; aria-pressed carries state. */
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
        size="sm"
        variant={checked ? 'solid' : 'ghost'}
        {...props}
        onClick={handleClick}
      >
        <Icon as={icon} boxSize="3.5" />
      </IconButton>
    </Tooltip>
  );
};

import type { ComponentProps, ReactNode, SyntheticEvent } from 'react';

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
   * as disabled (`aria-disabled`, described by the reason), lets no activation through (a submit button does not
   * submit), and shows the reason in its tooltip. The reason wins over `disabled`, which would hide it. Native
   * `disabled` alone stays for states the user can change, where there is nothing to explain.
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

/**
 * Cancels a click before any handler sees it, as native `disabled` would: the default (form submission, link
 * navigation) is prevented, and stopping propagation in the capture phase keeps the event from the control's own
 * handlers, a library's (Zag's menu item select), and its ancestors'. Keyboard activation arrives as a click too.
 */
const preventActivation = (event: SyntheticEvent) => {
  event.preventDefault();
  event.stopPropagation();
};

/**
 * What makes a Button or menu item unavailable: it reads as disabled (`aria-disabled`) yet stays focusable or
 * highlightable, no activation gets through, and a hidden copy of the reason describes it after any description the
 * caller gave. A tooltip cannot carry the description: a menu item is never focused, and a tooltip's own content exists
 * only while open. Render `description` inside the control. Callers drop native `disabled` while a reason holds, so the
 * reason is always reachable.
 */
export const useUnavailableControl = (reason: string | undefined, describedBy: string | undefined) => {
  const descriptionId = useId();

  if (!reason) {
    return null;
  }

  return {
    description: (
      <span hidden id={descriptionId}>
        {reason}
      </span>
    ),
    props: {
      'aria-describedby': describedBy ? `${describedBy} ${descriptionId}` : descriptionId,
      'aria-disabled': true,
      onClickCapture: preventActivation,
    } as const,
  };
};

/** Solid defaults to accent; other variants to gray. Explicit colorPalette overrides both. */
const defaultPalette = (variant: ButtonProps['variant']): ButtonProps['colorPalette'] =>
  variant === undefined || variant === 'solid' ? 'accent' : 'gray';

export const Button = ({ children, colorPalette, disabled, unavailableReason, ...props }: ButtonProps) => {
  const unavailable = useUnavailableControl(unavailableReason, props['aria-describedby']);
  const button = (
    <ChakraButton
      colorPalette={colorPalette ?? defaultPalette(props.variant)}
      {...props}
      {...unavailable?.props}
      disabled={unavailable ? undefined : disabled}
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
  disabled,
  tooltip,
  unavailableReason,
  ...props
}: IconButtonProps) => {
  const unavailable = useUnavailableControl(unavailableReason, props['aria-describedby']);
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
      disabled={unavailable ? undefined : disabled}
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

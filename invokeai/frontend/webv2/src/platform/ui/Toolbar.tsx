import type { ComponentProps, ReactNode } from 'react';

import { Icon, Separator, Stack, type StackProps } from '@chakra-ui/react';

import { IconButton } from './Button';

const TOOLBAR_TOOLTIP_PLACEMENT = 'right';

export const Toolbar = ({ children, direction = 'column', ...stackProps }: StackProps) => (
  <Stack
    aria-orientation={direction === 'column' ? 'vertical' : 'horizontal'}
    bg="bg.subtle"
    borderColor="border.subtle"
    borderWidth="1px"
    direction={direction}
    gap="0.5"
    p="1"
    role="toolbar"
    rounded="lg"
    shadow="sm"
    {...stackProps}
  >
    {children}
  </Stack>
);

export const ToolbarButton = ({
  icon,
  isActive,
  label,
  tooltip,
  ...buttonProps
}: {
  icon: React.ElementType;
  /** Sticky-tool styling; omit for one-shot action buttons. */
  isActive?: boolean;
  label: string;
  /** Richer tooltip content (a shortcut, a help line); the label alone by default. */
  tooltip?: ReactNode;
} & Omit<ComponentProps<typeof IconButton>, 'aria-label' | 'aria-labelledby' | 'tooltip' | 'tooltipPlacement'>) => (
  <IconButton
    aria-label={label}
    aria-pressed={isActive}
    tooltip={tooltip ?? label}
    tooltipPlacement={TOOLBAR_TOOLTIP_PLACEMENT}
    variant={isActive ? 'solid' : 'ghost'}
    {...buttonProps}
  >
    <Icon as={icon} boxSize="3.5" />
  </IconButton>
);

export const ToolbarSeparator = ({ direction = 'column' }: { direction?: 'column' | 'row' }) => (
  <Separator borderColor="border.subtle" orientation={direction === 'column' ? 'horizontal' : 'vertical'} />
);

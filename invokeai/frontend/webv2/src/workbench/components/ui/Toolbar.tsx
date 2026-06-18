import type { ComponentProps } from 'react';

import { Icon, Separator, Stack, type StackProps } from '@chakra-ui/react';

import { IconButton } from './Button';
import { Tooltip } from './Tooltip';

/**
 * The workbench's floating tool strip, shared by canvas-like surfaces (the
 * workflow editor today, the canvas later). Vertical by default — designed to
 * dock against a surface edge. Compose from `ToolbarButton`s (sticky tools via
 * `isActive`, one-shot actions without) and `ToolbarSeparator`s between
 * groups.
 */

export const Toolbar = ({ children, direction = 'column', ...stackProps }: StackProps) => (
  <Stack
    bg="bg.subtle"
    borderColor="border.subtle"
    borderWidth="1px"
    direction={direction}
    gap="0.5"
    p="1"
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
  ...buttonProps
}: {
  icon: React.ElementType;
  /** Sticky-tool styling; omit for one-shot action buttons. */
  isActive?: boolean;
  label: string;
} & Omit<ComponentProps<typeof IconButton>, 'aria-label'>) => (
  <Tooltip content={label} positioning={{ placement: 'right-start' }}>
    <IconButton
      aria-label={label}
      aria-pressed={isActive}
      size="sm"
      title={label}
      variant={isActive ? 'solid' : 'ghost'}
      {...buttonProps}
    >
      <Icon as={icon} boxSize="3.5" />
    </IconButton>
  </Tooltip>
);

export const ToolbarSeparator = ({ direction = 'column' }: { direction?: 'column' | 'row' }) => (
  <Separator borderColor="border.subtle" orientation={direction === 'column' ? 'horizontal' : 'vertical'} />
);

import type { ComponentProps } from 'react';

import { Icon, Separator, Stack, type StackProps } from '@chakra-ui/react';

import { IconButton } from './Button';
import { Tooltip } from './Tooltip';

const TOOLBAR_TOOLTIP_POSITIONING = { placement: 'right' } as const;

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
  ...buttonProps
}: {
  icon: React.ElementType;
  /** Sticky-tool styling; omit for one-shot action buttons. */
  isActive?: boolean;
  label: string;
} & Omit<ComponentProps<typeof IconButton>, 'aria-label' | 'aria-labelledby'>) => (
  <Tooltip content={label} positioning={TOOLBAR_TOOLTIP_POSITIONING}>
    <IconButton
      aria-label={label}
      aria-pressed={isActive}
      size="xs"
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

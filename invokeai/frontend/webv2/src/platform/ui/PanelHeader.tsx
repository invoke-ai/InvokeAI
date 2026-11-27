import type { ElementType, ReactNode } from 'react';

import { Box, HStack, Icon, type StackProps } from '@chakra-ui/react';
import { useCallback, useMemo } from 'react';

import { IconButton } from './Button';

/** Shared desktop panel chrome; callers supply identity and surface-specific actions. */
export const PanelHeader = (props: StackProps) => (
  <HStack
    justify="space-between"
    borderBottomWidth="1px"
    borderColor="border.subtle"
    h="10"
    minH="10"
    ps="3"
    pe="2"
    {...props}
  />
);

/** A thin vertical rule between groups of header actions. */
export const HeaderActionDivider = () => <Box bg="border.subtle" flexShrink={0} h="4" mx="1" w="1px" />;

/**
 * An undo or redo step among a header's actions. With nothing to step over it reads as disabled (`aria-disabled`) but
 * keeps focus, so stepping to the end of the history does not drop a keyboard user to the page; it never runs then.
 * `unavailableReason` says why a step that exists cannot run now, in the tooltip and the description.
 */
export const HeaderHistoryButton = ({
  available,
  icon,
  label,
  shortcut,
  unavailableReason,
  onStep,
}: {
  available: boolean;
  icon: ElementType;
  /** The accessible name and the tooltip's text. */
  label: string;
  /** The step's keycaps, after the label in the tooltip; the caller renders them, since key glyphs are not platform. */
  shortcut?: ReactNode;
  unavailableReason?: string;
  onStep: () => void;
}) => {
  // The click event must not reach a step that takes an optional argument.
  const onClick = useCallback(() => {
    if (available && !unavailableReason) {
      onStep();
    }
  }, [available, onStep, unavailableReason]);
  const tooltip = useMemo(
    () =>
      shortcut ? (
        <HStack gap="2">
          <span>{label}</span>
          {shortcut}
        </HStack>
      ) : (
        label
      ),
    [label, shortcut]
  );

  return (
    <IconButton
      aria-disabled={available ? undefined : true}
      aria-label={label}
      color="fg.muted"
      size="sm"
      tooltip={tooltip}
      unavailableReason={unavailableReason}
      variant="ghost"
      onClick={onClick}
    >
      <Icon as={icon} boxSize="3.5" />
    </IconButton>
  );
};

import type { GalleryBoardSectionId } from '@features/gallery/core/settings';

import { Collapsible, HStack, Stack, Text } from '@chakra-ui/react';
import { ChevronRightIcon } from 'lucide-react';
import { useCallback, type ReactNode } from 'react';

const INDICATOR_OPEN_STYLES = { transform: 'rotate(90deg)' } as const;
const TRIGGER_HOVER_STYLES = { color: 'fg' } as const;

/**
 * One collapsible group in the board panel. The trailing `action` is a sibling
 * of the disclosure trigger rather than a child, so the "+" button is not a
 * <button> nested inside another one.
 */
export const GalleryBoardSection = ({
  action,
  children,
  isOpen,
  label,
  sectionId,
  onToggle,
}: {
  action?: ReactNode;
  children: ReactNode;
  isOpen: boolean;
  label: string;
  sectionId: GalleryBoardSectionId;
  onToggle: (sectionId: GalleryBoardSectionId, isOpen: boolean) => void;
}) => {
  const handleOpenChange = useCallback(
    ({ open }: { open: boolean }) => onToggle(sectionId, open),
    [sectionId, onToggle]
  );

  return (
    <Collapsible.Root open={isOpen} onOpenChange={handleOpenChange}>
      <HStack gap="1" minH="6" pe="1" ps="1">
        <Collapsible.Trigger
          alignItems="center"
          color="fg.muted"
          display="flex"
          flex="1"
          gap="1"
          minW="0"
          transition="color var(--wb-motion-duration-fast) ease"
          _hover={TRIGGER_HOVER_STYLES}
        >
          <Collapsible.Indicator _open={INDICATOR_OPEN_STYLES} transition="transform var(--wb-motion-duration-medium)">
            <ChevronRightIcon size="12" />
          </Collapsible.Indicator>
          <Text
            as="span"
            fontSize="2xs"
            fontWeight="600"
            letterSpacing="wide"
            lineHeight="1"
            textTransform="uppercase"
            truncate
          >
            {label}
          </Text>
        </Collapsible.Trigger>
        {action}
      </HStack>
      <Collapsible.Content>
        <Stack gap="0.5" pb="1">
          {children}
        </Stack>
      </Collapsible.Content>
    </Collapsible.Root>
  );
};

import { Box, Collapsible, Flex, Text } from '@chakra-ui/react';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { ChevronRightIcon } from 'lucide-react';
import { useCallback } from 'react';

type Props = {
  label: string;
  isOpen?: boolean;
  defaultOpen?: boolean;
  /** When set, the open state persists per user across reloads under this id. */
  sectionId?: string;
  badges?: React.ReactNode;
  children: React.ReactNode;
};

const COLLAPSIBLE_INDICATOR_OPEN_STYLES = { transform: 'rotate(90deg)' };

export const GenerateCollapsibleSection = ({ badges, children, defaultOpen, isOpen, label, sectionId }: Props) => {
  const { sectionPreferences } = useGenerationUi();
  const persistedOpen = sectionId === undefined ? undefined : sectionPreferences.sectionsOpen[sectionId];
  const resolvedOpen = isOpen ?? (sectionId === undefined ? undefined : (persistedOpen ?? defaultOpen ?? false));
  const handleOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      if (sectionId !== undefined) {
        sectionPreferences.setSectionOpen(sectionId, open);
      }
    },
    [sectionId, sectionPreferences]
  );

  return (
    <Collapsible.Root
      bg="bg.muted/50"
      defaultOpen={sectionId === undefined ? defaultOpen : undefined}
      open={resolvedOpen}
      overflow="hidden"
      rounded="md"
      onOpenChange={sectionId === undefined ? undefined : handleOpenChange}
    >
      <Collapsible.Trigger display="flex" gap={2} w="full" px={2} h="8" alignItems="center">
        <Collapsible.Indicator
          _open={COLLAPSIBLE_INDICATOR_OPEN_STYLES}
          transition="transform var(--wb-motion-duration-slow)"
        >
          <ChevronRightIcon size="14" />
        </Collapsible.Indicator>
        <Text
          as="span"
          fontSize="2xs"
          truncate
          letterSpacing="widest"
          fontWeight="bold"
          textTransform="uppercase"
          color="fg.muted"
          lineHeight="1"
        >
          {label}
        </Text>

        <Flex gap={1} ml="auto" fontFamily="mono">
          {badges}
        </Flex>
      </Collapsible.Trigger>
      <Collapsible.Content bg="bg.muted">
        <Box borderTopWidth={1} borderColor="bg.subtle">
          {children}
        </Box>
      </Collapsible.Content>
    </Collapsible.Root>
  );
};

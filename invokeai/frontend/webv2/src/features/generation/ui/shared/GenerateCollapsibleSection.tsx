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

/** Style uncontrolled sections from DOM state; preserve geometry and draw one overlay separator per boundary. */
const SECTION_OPEN_STYLES = { bg: 'bg.muted', rounded: 'sm' };
const SECTION_STYLES = {
  '&::before': {
    content: '""',
    position: 'absolute',
    insetInline: 0,
    top: 'calc(var(--chakra-spacing-1) / -2)',
    borderTopWidth: '1px',
    pointerEvents: 'none',
    opacity: 0,
    transition: 'opacity var(--wb-motion-duration-slow) ease',
  },
  '.generate-section + &::before': { opacity: 1 },
  // With equal specificity, open-state rules must follow preceding-section rules.
  '&[data-state="open"]::before': { opacity: 0 },
  '.generate-section[data-state="open"] + &::before': { opacity: 0 },
} as const;

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
      className="generate-section"
      css={SECTION_STYLES}
      _open={SECTION_OPEN_STYLES}
      defaultOpen={sectionId === undefined ? defaultOpen : undefined}
      open={resolvedOpen}
      position="relative"
      transition="background var(--wb-motion-duration-slow) ease, border-radius var(--wb-motion-duration-slow) ease"
      onOpenChange={sectionId === undefined ? undefined : handleOpenChange}
    >
      <Collapsible.Trigger display="flex" gap={2} w="full" px={1.5} h="8" alignItems="center">
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

      <Collapsible.Content>
        <Box borderTopWidth={1} borderColor="bg.subtle">
          {children}
        </Box>
      </Collapsible.Content>
    </Collapsible.Root>
  );
};

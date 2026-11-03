import { Box, chakra, Icon, Stack, Text } from '@chakra-ui/react';
import { ChevronRightIcon } from 'lucide-react';
import { useCallback, useId, type ReactNode } from 'react';

const TOGGLE_HOVER_STYLES = { color: 'fg' } as const;
const TOGGLE_DISABLED_STYLES = { cursor: 'default' } as const;

/**
 * One project's boards under Other projects, behind a disclosure named by the project. While the board search has
 * text the group shows its matches whatever its saved state, so a search never hides one; the saved state is left
 * as it is. A collapsed group renders no rows: the boards panel is not virtualized.
 */
export const GalleryProjectGroup = ({
  children,
  isCollapsed,
  isSearching,
  label,
  projectId,
  onToggle,
}: {
  children: ReactNode;
  isCollapsed: boolean;
  isSearching: boolean;
  label: string;
  projectId: string;
  onToggle: (projectId: string, isOpen: boolean) => void;
}) => {
  const contentId = useId();
  const isOpen = isSearching || !isCollapsed;
  const handleClick = useCallback(() => onToggle(projectId, !isOpen), [isOpen, onToggle, projectId]);

  return (
    <Stack gap="0.5">
      <Box aria-level={4} role="heading">
        <chakra.button
          alignItems="center"
          aria-controls={contentId}
          aria-expanded={isOpen}
          color="fg.muted"
          data-gallery-project-group={projectId}
          disabled={isSearching}
          display="flex"
          gap="1"
          minW="0"
          pe="2"
          ps="2"
          pt="1"
          rounded="sm"
          textAlign="start"
          transition="color var(--wb-motion-duration-fast) ease"
          type="button"
          w="full"
          _disabled={TOGGLE_DISABLED_STYLES}
          _hover={TOGGLE_HOVER_STYLES}
          onClick={handleClick}
        >
          <Icon
            as={ChevronRightIcon}
            boxSize="3"
            flexShrink={0}
            transform={isOpen ? 'rotate(90deg)' : undefined}
            transition="transform var(--wb-motion-duration-medium)"
          />
          <Text as="span" fontSize="xs" fontWeight="600" truncate>
            {label}
          </Text>
        </chakra.button>
      </Box>
      <Stack display={isOpen ? undefined : 'none'} gap="0.5" id={contentId}>
        {isOpen ? children : null}
      </Stack>
    </Stack>
  );
};

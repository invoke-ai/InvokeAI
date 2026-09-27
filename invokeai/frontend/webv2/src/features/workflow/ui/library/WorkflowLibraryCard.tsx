import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';

import { Badge, Box, Flex, HStack, Icon, Image, Skeleton, Stack, Text } from '@chakra-ui/react';
import { getModelBaseLabel } from '@features/models';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { ImageOffIcon } from 'lucide-react';
import { useCallback, useState, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Select on click, open on double-click, and select/context-menu on right-click. Reserve enrichment height so
 * async facts cannot reflow the grid.
 */

const CARD_HOVER = { bg: 'bg.muted', borderColor: 'border.emphasized' } as const;
const CARD_FOCUS_VISIBLE = { outline: '2px solid {colors.accent.solid}', outlineOffset: '-2px' } as const;
const CARD_TRANSITION =
  'border-color var(--wb-motion-duration-medium) ease, background var(--wb-motion-duration-medium) ease';
const THUMBNAIL_ASPECT_RATIO = 3 / 2;

/** The card's DOM id, which the rail's context menu names as its trigger so it nests under the dialog's layer. */
export const getWorkflowLibraryCardId = (workflowId: string): string => `workflow-library-card-${workflowId}`;

export interface WorkflowLibraryCardProps {
  entry: WorkflowLibraryEntry;
  /** The project's active workflow; the card says so in the header. */
  isActive?: boolean;
  isSelected: boolean;
  /** Models this workflow needs that are not installed; 0 hides the badge. */
  missingCount: number;
  onContextMenu: (workflowId: string, point: { x: number; y: number }) => void;
  onOpen: (workflowId: string) => void;
  onSelect: (workflowId: string) => void;
}

export const WorkflowLibraryCard = ({
  entry,
  isActive = false,
  isSelected,
  missingCount,
  onContextMenu,
  onOpen,
  onSelect,
}: WorkflowLibraryCardProps) => {
  const { t } = useTranslation();
  const [hasThumbnailFailed, setHasThumbnailFailed] = useState(false);
  const { enrichment, item } = entry;
  const workflowId = item.workflow_id;

  const handleSelect = useCallback(() => onSelect(workflowId), [onSelect, workflowId]);
  const handleOpen = useCallback(() => onOpen(workflowId), [onOpen, workflowId]);
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      event.preventDefault();

      // A keyboard-raised menu (Shift+F10, the Menu key) carries no pointer
      // position; anchor it inside the card instead of at the viewport corner.
      const rect = event.currentTarget.getBoundingClientRect();
      const fromKeyboard = event.clientX === 0 && event.clientY === 0;

      onContextMenu(
        workflowId,
        fromKeyboard ? { x: rect.left + 16, y: rect.top + 16 } : { x: event.clientX, y: event.clientY }
      );
    },
    [onContextMenu, workflowId]
  );
  const handleThumbnailError = useCallback(() => setHasThumbnailFailed(true), []);

  const showThumbnail = Boolean(item.thumbnail_url) && !hasThumbnailFailed;
  const primaryBase = enrichment.status === 'ready' ? enrichment.requirements.primaryBase : null;

  return (
    <Box
      as="button"
      aria-pressed={isSelected}
      bg={isSelected ? 'bg.emphasized' : 'bg.subtle'}
      borderColor={isSelected ? 'accent.solid' : 'border.subtle'}
      borderWidth="1px"
      data-workflow-card={workflowId}
      id={getWorkflowLibraryCardId(workflowId)}
      minW="0"
      overflow="hidden"
      rounded="lg"
      textAlign="start"
      transition={CARD_TRANSITION}
      w="full"
      _focusVisible={CARD_FOCUS_VISIBLE}
      _hover={CARD_HOVER}
      onClick={handleSelect}
      onContextMenu={handleContextMenu}
      onDoubleClick={handleOpen}
    >
      <Box aspectRatio={THUMBNAIL_ASPECT_RATIO} bg="bg.muted" overflow="hidden" w="full">
        {showThumbnail ? (
          <Image
            alt=""
            h="full"
            objectFit="cover"
            src={item.thumbnail_url ?? undefined}
            w="full"
            onError={handleThumbnailError}
          />
        ) : (
          <Flex align="center" direction="column" gap="1" h="full" justify="center" w="full">
            <Icon aria-hidden as={ImageOffIcon} boxSize="5" color="fg.subtle" opacity={0.6} />
            <Text color="fg.subtle" fontSize="2xs">
              {t('workflowLibrary.notRunYet')}
            </Text>
          </Flex>
        )}
      </Box>
      <Stack gap="1" minW="0" p="2.5" w="full">
        <HStack gap="1.5" minW="0">
          <MiddleTruncate fontSize="xs" fontWeight="600" minW="0" text={item.name || t('workflowLibrary.untitled')} />
          {isActive ? (
            <Badge data-active-workflow flexShrink={0} size="xs" variant="solid">
              {t('workflowLibrary.activeWorkflow')}
            </Badge>
          ) : null}
        </HStack>
        <HStack gap="1.5" h="4" minW="0">
          {enrichment.status === 'pending' ? (
            // Show placeholders only while enriching; unreadable workflows leave facts absent.
            <Skeleton data-enrichment-placeholder h="3" rounded="sm" w="14" />
          ) : null}
          {primaryBase ? (
            <Badge flexShrink={0} size="xs" variant="subtle">
              {getModelBaseLabel(primaryBase)}
            </Badge>
          ) : null}
          {enrichment.status === 'ready' ? (
            <Text color="fg.muted" fontSize="2xs" truncate>
              {t('workflowLibrary.nodeCount', { count: enrichment.nodeCount })}
            </Text>
          ) : null}
          {missingCount > 0 ? (
            <Badge bg="bg.warning" color="fg.warning" flexShrink={0} size="xs" variant="subtle">
              {t('workflowLibrary.installModels', { count: missingCount })}
            </Badge>
          ) : null}
        </HStack>
      </Stack>
    </Box>
  );
};

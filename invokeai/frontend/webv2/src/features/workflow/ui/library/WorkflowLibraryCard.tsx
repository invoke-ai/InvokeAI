import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';

import { Badge, Box, Flex, HStack, Icon, Image, Skeleton, Stack, Text } from '@chakra-ui/react';
import { getModelBaseLabel } from '@features/models';
import { IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Tooltip } from '@platform/ui/Tooltip';
import { EllipsisIcon, ImageOffIcon } from 'lucide-react';
import { useCallback, useState, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Select on click, open on double-click, and select/context-menu on right-click or from the title's own menu
 * button. Reserve enrichment height so async facts cannot reflow the grid.
 */

const CARD_HOVER = { bg: 'bg.muted', borderColor: 'border.emphasized' } as const;
const CARD_FOCUS_VISIBLE = { outline: '2px solid {colors.accent.solid}', outlineOffset: '-2px' } as const;
const CARD_TRANSITION =
  'border-color var(--wb-motion-duration-medium) ease, background var(--wb-motion-duration-medium) ease';
const THUMBNAIL_ASPECT_RATIO = 3 / 2;

/** The card's DOM id, which the rail's context menu names as its trigger so it nests under the dialog's layer. */
export const getWorkflowLibraryCardId = (workflowId: string): string => `workflow-library-card-${workflowId}`;

/** The title's menu button; when it opened the menu, it is the trigger focus returns to. */
export const getWorkflowLibraryCardMenuId = (workflowId: string): string =>
  `${getWorkflowLibraryCardId(workflowId)}-menu`;

/**
 * The rail's actions menu content, named so the tile button can point at it: the dialog's focus trap only lets
 * focus into portalled content that an expanded control inside the dialog names with aria-controls.
 */
export const getWorkflowLibraryCardMenuContentId = (workflowId: string): string =>
  `${getWorkflowLibraryCardId(workflowId)}-actions`;

/** What opened a card's actions menu: a pointer position on the card, or the tile's own menu button. */
export type WorkflowCardMenuAnchor =
  | { kind: 'point'; workflowId: string; x: number; y: number }
  | { kind: 'trigger'; workflowId: string; triggerId: string };

/**
 * A right-click on a card, or a click on a tile's menu button, re-anchors or toggles the open menu through its own
 * handler rather than dismissing it as an outside press.
 */
export const keepCardMenuOpenForRetarget = (event: {
  detail: { originalEvent: Event };
  preventDefault(): void;
}): void => {
  const original = event.detail.originalEvent;

  if (!(original instanceof PointerEvent) || !(original.target instanceof Element)) {
    return;
  }

  if (
    (original.button === 2 && original.target.closest('[data-workflow-card]') !== null) ||
    (original.button === 0 && original.target.closest('[data-workflow-card-menu]') !== null)
  ) {
    event.preventDefault();
  }
};

/** The title row's fixed height; the sibling menu button sits on it from outside the card button. */
const TITLE_ROW_HEIGHT = '6';
/** Body padding (2.5) plus the facts row (4) plus the gap (1) below the title row, in px. */
const TITLE_ROW_BOTTOM_PX = 30;

export interface WorkflowLibraryCardProps {
  entry: WorkflowLibraryEntry;
  /** The project's active workflow; the card says so in the header. */
  isActive?: boolean;
  /** Which of the tile's controls has the actions menu open, if any; that control names the menu for the dialog. */
  menuOpenedBy?: 'card' | 'button' | null;
  isSelected: boolean;
  /** Models this workflow needs that are not installed; 0 hides the badge. */
  missingCount: number;
  onContextMenu: (workflowId: string, anchor: WorkflowCardMenuAnchor) => void;
  onOpen: (workflowId: string) => void;
  onSelect: (workflowId: string) => void;
}

export const WorkflowLibraryCard = ({
  entry,
  isActive = false,
  isSelected,
  menuOpenedBy = null,
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
        fromKeyboard
          ? { kind: 'point', workflowId, x: rect.left + 16, y: rect.top + 16 }
          : { kind: 'point', workflowId, x: event.clientX, y: event.clientY }
      );
    },
    [onContextMenu, workflowId]
  );
  const handleMenuButton = useCallback(
    () =>
      onContextMenu(workflowId, { kind: 'trigger', triggerId: getWorkflowLibraryCardMenuId(workflowId), workflowId }),
    [onContextMenu, workflowId]
  );
  const handleThumbnailError = useCallback(() => setHasThumbnailFailed(true), []);

  const showThumbnail = Boolean(item.thumbnail_url) && !hasThumbnailFailed;
  const primaryBase = enrichment.status === 'ready' ? enrichment.requirements.primaryBase : null;

  const menuContentId = getWorkflowLibraryCardMenuContentId(workflowId);

  // The menu button is a sibling of the card button, never nested in it: a button cannot hold another control.
  return (
    <Box minW="0" position="relative" w="full">
      <Box
        as="button"
        aria-controls={menuOpenedBy === 'card' ? menuContentId : undefined}
        aria-expanded={menuOpenedBy === 'card' ? true : undefined}
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
              <Text color="fg.subtle" fontSize="xs">
                {t('workflowLibrary.notRunYet')}
              </Text>
            </Flex>
          )}
        </Box>
        <Stack gap="1" minW="0" p="2.5" w="full">
          <HStack gap="1.5" h={TITLE_ROW_HEIGHT} minW="0" pe="7">
            <MiddleTruncate fontSize="md" fontWeight="600" minW="0" text={item.name || t('workflowLibrary.untitled')} />
            {isActive ? (
              <Badge data-active-workflow flexShrink={0} variant="solid">
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
              <Badge flexShrink={0} variant="subtle">
                {getModelBaseLabel(primaryBase)}
              </Badge>
            ) : null}
            {enrichment.status === 'ready' ? (
              <Text color="fg.muted" fontSize="xs" truncate>
                {t('workflowLibrary.nodeCount', { count: enrichment.nodeCount })}
              </Text>
            ) : null}
            {missingCount > 0 ? (
              <Badge bg="bg.warning" color="fg.warning" flexShrink={0} variant="subtle">
                {t('workflowLibrary.installModels', { count: missingCount })}
              </Badge>
            ) : null}
          </HStack>
        </Stack>
      </Box>
      <Tooltip content={t('workflowLibrary.moreActions')}>
        <IconButton
          aria-controls={menuOpenedBy === 'button' ? menuContentId : undefined}
          aria-expanded={menuOpenedBy === 'button'}
          aria-haspopup="menu"
          aria-label={t('workflowLibrary.moreActions')}
          bottom={`${TITLE_ROW_BOTTOM_PX}px`}
          color="fg.muted"
          data-workflow-card-menu={workflowId}
          id={getWorkflowLibraryCardMenuId(workflowId)}
          insetEnd="2.5"
          position="absolute"
          size="sm"
          variant="ghost"
          onClick={handleMenuButton}
        >
          <EllipsisIcon />
        </IconButton>
      </Tooltip>
    </Box>
  );
};

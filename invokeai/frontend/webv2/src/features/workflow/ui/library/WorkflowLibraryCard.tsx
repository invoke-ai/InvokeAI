import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';

import { Badge, Box, HStack, Icon, Stack, Text, type SystemStyleObject } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { ClippedTooltip, Tooltip } from '@platform/ui/Tooltip';
import { EllipsisIcon, FolderCheckIcon, WorkflowIcon } from 'lucide-react';
import { memo, use, useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { formatTimeSince, useDataTimeSnapshot } from './relativeTime';
import { getWorkflowLibraryEntryFacts, WorkflowLibraryFactsLine } from './WorkflowLibraryFacts';
import { WorkflowThumbnailPlaceholder } from './WorkflowLibraryThumbnail';
import {
  WORKFLOW_THUMBNAIL_ASPECT_RATIO,
  WorkflowLibraryCardSizeContext,
  WorkflowThumbnailImage,
} from './workflowThumbnail';

/**
 * A library card is presentation only: the grid handles its click (select), double-click (open), right-click and menu
 * button (its own actions menu, which leaves the selection alone) and quick actions by delegation from the data attributes below, so a card renders again only
 * when its own entry or state changes. Reserve enrichment height so async facts cannot reflow the grid. Pointer hover
 * (and focus within the card) reveals quick actions over the top of the image: Open and Preview are pointer
 * conveniences outside the tab order that never take focus (the grid hands it to the card on press), and the menu
 * button beside them is the card's keyboard-reachable control.
 */

const CARD_HOVER = { bg: 'bg.muted', borderColor: 'border.emphasized' } as const;
/** The selected card keeps its accent border under the pointer; only its fill moves one step. */
const SELECTED_CARD_HOVER = { bg: 'bg.muted', borderColor: 'accent.solid' } as const;
const CARD_FOCUS_VISIBLE = { outline: '2px solid {colors.accent.solid}', outlineOffset: '-2px' } as const;
const CARD_TRANSITION =
  'border-color var(--wb-motion-duration-medium) ease, background var(--wb-motion-duration-medium) ease';

/**
 * Quick actions show while the pointer is over the card, focus is inside it, or its menu is open, and take the pointer
 * only while shown: a transparent control would swallow a press meant for the card. Without hover (touch), the
 * pointer-only actions never show and the menu button always does, so a tap on the image still selects.
 */
const QUICK_ACTION_SHOWN = { opacity: 1, pointerEvents: 'auto' } as const;
const CARD_ROOT_CSS: SystemStyleObject = {
  '& [data-card-quick-action]': {
    opacity: 0,
    pointerEvents: 'none',
    transition: 'opacity var(--wb-motion-duration-fast) ease',
  },
  '&:focus-within [data-card-quick-action], &[data-menu-open] [data-card-quick-action]': QUICK_ACTION_SHOWN,
  '@media (hover: hover)': { '&:hover [data-card-quick-action]': QUICK_ACTION_SHOWN },
  '@media (hover: none)': {
    '& [data-card-quick-action][data-pointer-only]': { display: 'none' },
    '& [data-card-quick-action]': QUICK_ACTION_SHOWN,
  },
};

/** The card's DOM id, which its actions menu names as its trigger so it nests under the dialog's layer. */
export const getWorkflowLibraryCardId = (workflowId: string): string => `workflow-library-card-${workflowId}`;

/** The card's menu button; when it opened the menu, it is the trigger focus returns to. */
export const getWorkflowLibraryCardMenuId = (workflowId: string): string =>
  `${getWorkflowLibraryCardId(workflowId)}-menu`;

/**
 * The card's actions menu content, named so the tile button can point at it: the dialog's focus trap only lets
 * focus into portalled content that an expanded control inside the dialog names with aria-controls.
 */
export const getWorkflowLibraryCardMenuContentId = (workflowId: string): string =>
  `${getWorkflowLibraryCardId(workflowId)}-actions`;

/** What opened a card's actions menu: a pointer position on the card, or the tile's own menu button. */
export type WorkflowCardMenuAnchor =
  | { kind: 'point'; workflowId: string; x: number; y: number }
  | { kind: 'trigger'; workflowId: string; triggerId: string };

export type WorkflowCardContextMenuHandler = (workflowId: string, anchor: WorkflowCardMenuAnchor) => void;

/** A quick action's `data-card-action`; the grid reads it with the card root's `data-workflow-card-root`. */
export type WorkflowCardQuickAction = 'open' | 'preview';

/**
 * A right-click on a card (or its quick actions) or a list row, or a click on a tile's or row's menu button, re-anchors
 * or toggles the open menu through its own handler rather than dismissing it as an outside press.
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
    (original.button === 2 &&
      original.target.closest(
        '[data-workflow-card],[data-workflow-card-actions],[data-workflow-library-list] [data-list-row]'
      ) !== null) ||
    (original.button === 0 && original.target.closest('[data-workflow-card-menu]') !== null)
  ) {
    event.preventDefault();
  }
};

/**
 * Preview and More sit on an opaque panel fill so they read over any thumbnail. That fill replaces the outline recipe's
 * translucent hover and expanded tint, so they lay the same tint over the panel (`bg.muted` equals `bg.panel`).
 */
const QUICK_ACTION_INTERACTION = { bg: 'color-mix(in oklab, {colors.gray.hoverTint} 10%, {colors.bg.panel})' } as const;

/**
 * The library's menus (a card's, the rail's More, the sheet's More) are siblings, never nested. Zag handles an outside
 * press a frame after its pointerdown, so a quick press on one menu's trigger while another is open opens the new menu
 * first; the old one then leaves the layer stack with the new one above it and would dismiss it as its nested layer.
 * A dismissal requested by a closing menu is ignored; a closing dialog still dismisses its menus.
 */
export const keepMenuOpenWhenSiblingMenuCloses = (event: {
  detail: { targetLayer: HTMLElement | undefined };
  preventDefault(): void;
}): void => {
  if (event.detail.targetLayer?.matches('[data-scope="menu"][data-part="content"]')) {
    event.preventDefault();
  }
};

const NAME_FONT_SIZE = { large: 'lg', small: 'md' } as const;
const QUICK_ACTION_SIZE = { large: 'sm', small: 'xs' } as const;

export interface WorkflowLibraryCardProps {
  entry: WorkflowLibraryEntry;
  /** The project's active workflow; the card says so in the header. */
  isActive?: boolean;
  /** The project already holds a copy of this library template. */
  isInProject?: boolean;
  /** Which of the tile's controls has the actions menu open, if any; that control names the menu for the dialog. */
  menuOpenedBy?: 'card' | 'button' | null;
  isSelected: boolean;
  /** Multi-user servers badge public templates; project workflows have no library visibility. */
  showVisibilityStatus?: boolean;
  /** Project workflows say when they last ran instead of their node count. */
  showLastRun?: boolean;
  /** Models this workflow needs that are not installed; 0 when none. */
  missingCount: number;
  /** Node packs the workflow uses that this server lacks; null while node templates are unknown. */
  missingNodePackCount?: number | null;
  /** Shows the Preview quick action. */
  canPreview?: boolean;
}

const isTextClipped = (element: HTMLElement | null): boolean =>
  element !== null && element.scrollWidth > element.clientWidth;

/** Memoised: every enrichment completion and range load re-renders the grid, but rarely a given card's props. */
export const WorkflowLibraryCard = memo(function WorkflowLibraryCard({
  canPreview = false,
  entry,
  isActive = false,
  isInProject = false,
  isSelected,
  menuOpenedBy = null,
  missingCount,
  missingNodePackCount = null,
  showLastRun = false,
  showVisibilityStatus = false,
}: WorkflowLibraryCardProps) {
  const { t } = useTranslation();
  const size = use(WorkflowLibraryCardSizeContext);
  const [hasThumbnailFailed, setHasThumbnailFailed] = useState(false);
  const nameRef = useRef<HTMLParagraphElement>(null);
  const { enrichment, item } = entry;
  const workflowId = item.workflow_id;
  const dataTime = useDataTimeSnapshot(showLastRun ? item : null);

  // Pointer-only parts (Open, Preview, the name's tooltip) mount on the first pointer visit: a row of cards mounts in
  // one scroll frame, and these controls would roughly double what each card costs to mount.
  const [isPointerArmed, setIsPointerArmed] = useState(false);
  const armPointer = useCallback(() => setIsPointerArmed(true), []);
  const handleThumbnailError = useCallback(() => setHasThumbnailFailed(true), []);
  const isNameClipped = useCallback(() => isTextClipped(nameRef.current), []);

  const showThumbnail = Boolean(item.thumbnail_url) && !hasThumbnailFailed;
  const ready = enrichment.status === 'ready' ? enrichment : null;
  const lastRun = showLastRun && item.last_run_at ? formatTimeSince(item.last_run_at, dataTime) : '';
  const facts = getWorkflowLibraryEntryFacts({ entry, isInProject, lastRun, missingCount, missingNodePackCount }, t);
  const name = item.name || t('workflowLibrary.untitled');
  const menuContentId = getWorkflowLibraryCardMenuContentId(workflowId);
  const quickActionSize = QUICK_ACTION_SIZE[size];
  const nameText = (
    <Text ref={nameRef} flex="1" fontSize={NAME_FONT_SIZE[size]} fontWeight="600" minW="0" truncate>
      {name}
    </Text>
  );

  // The quick actions and menu button are siblings of the card button, never nested in it: a button cannot hold
  // another control.
  return (
    <Box
      css={CARD_ROOT_CSS}
      data-workflow-card-root={workflowId}
      data-menu-open={menuOpenedBy !== null ? '' : undefined}
      onPointerEnter={isPointerArmed ? undefined : armPointer}
      minW="0"
      position="relative"
      w="full"
    >
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
        _hover={isSelected ? SELECTED_CARD_HOVER : CARD_HOVER}
      >
        <Box aspectRatio={WORKFLOW_THUMBNAIL_ASPECT_RATIO} bg="bg.muted" overflow="hidden" position="relative" w="full">
          {showThumbnail ? (
            <WorkflowThumbnailImage src={item.thumbnail_url} onError={handleThumbnailError} />
          ) : (
            <WorkflowThumbnailPlaceholder workflowDocument={ready?.document ?? null} />
          )}
          {isInProject ? (
            <HStack
              bg="bg.panel/85"
              color="fg.muted"
              data-workflow-in-project
              insetStart="1.5"
              position="absolute"
              px="1"
              py="0.5"
              rounded="sm"
              bottom="1.5"
            >
              <Icon aria-hidden as={FolderCheckIcon} boxSize="3" />
              <Text srOnly>{t('workflowLibrary.inProject')}</Text>
            </HStack>
          ) : null}
        </Box>
        <Stack gap="1" minW="0" p="2.5" w="full">
          <HStack gap="1.5" h="6" minW="0">
            {isPointerArmed ? (
              <ClippedTooltip content={name} isClipped={isNameClipped}>
                {nameText}
              </ClippedTooltip>
            ) : (
              nameText
            )}
            {showVisibilityStatus && item.is_public === true ? (
              <Badge data-workflow-visibility="public" flexShrink={0} variant="subtle">
                {t('workflowLibrary.public')}
              </Badge>
            ) : null}
            {isActive ? (
              <Badge data-active-workflow flexShrink={0} variant="solid">
                {t('workflowLibrary.activeWorkflow')}
              </Badge>
            ) : null}
          </HStack>
          <WorkflowLibraryFactsLine facts={facts} />
        </Stack>
      </Box>
      <HStack
        // Over the top of the image, clear of the card's centre (there a press selects the card, and a strip the hover
        // revealed would take it instead) and of the in-project marker at the image's bottom corner.
        top="1.5"
        data-workflow-card-actions={workflowId}
        gap="1"
        insetEnd="1.5"
        insetStart="1.5"
        justify="flex-end"
        pointerEvents="none"
        position="absolute"
      >
        {/* The active project workflow is already open; its rail disables Open too. */}
        {isPointerArmed && !isActive ? (
          <Button
            aria-hidden
            data-card-action={'open' satisfies WorkflowCardQuickAction}
            data-card-quick-action
            data-pointer-only
            size={quickActionSize}
            tabIndex={-1}
            variant="solid"
          >
            {/* Parity with the rail: opening asks what to do once the project holds a copy. */}
            {isInProject ? t('workflowLibrary.openWithEllipsis') : t('workflowLibrary.open')}
          </Button>
        ) : null}
        {isPointerArmed && canPreview ? (
          <Tooltip content={t('workflowLibrary.preview.actionHint')}>
            <IconButton
              aria-hidden
              aria-label={t('workflowLibrary.preview.action')}
              bg="bg.panel"
              _hover={QUICK_ACTION_INTERACTION}
              data-card-action={'preview' satisfies WorkflowCardQuickAction}
              data-card-quick-action
              data-pointer-only
              size={quickActionSize}
              tabIndex={-1}
              variant="outline"
            >
              <WorkflowIcon />
            </IconButton>
          </Tooltip>
        ) : null}
        {/* Always mounted with its own tooltip: it is the card's keyboard control (Tab order, focus restore and the
            grid's Tab handling find it by id), and wrapping it later would remount it under focus. */}
        <Tooltip content={t('workflowLibrary.moreActions')}>
          <IconButton
            aria-controls={menuOpenedBy === 'button' ? menuContentId : undefined}
            aria-expanded={menuOpenedBy === 'button'}
            aria-haspopup="menu"
            aria-label={t('workflowLibrary.moreActions')}
            bg="bg.panel"
            _expanded={QUICK_ACTION_INTERACTION}
            _hover={QUICK_ACTION_INTERACTION}
            data-card-quick-action
            data-workflow-card-menu={workflowId}
            id={getWorkflowLibraryCardMenuId(workflowId)}
            size={quickActionSize}
            variant="outline"
          >
            <EllipsisIcon />
          </IconButton>
        </Tooltip>
      </HStack>
    </Box>
  );
});

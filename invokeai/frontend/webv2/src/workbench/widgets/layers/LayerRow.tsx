import type { CanvasLayerContract, CanvasNodeContract, SemanticNode } from '@workbench/canvas-engine/api';
import type { CSSProperties, FocusEvent, KeyboardEvent, MouseEvent } from 'react';

import { Badge, Box, chakra, HStack, Icon, Input, Text } from '@chakra-ui/react';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import { IconButton, Tooltip } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { isGroupNode, isHideableLayer, isNodeHidden, isOverlayStack } from '@workbench/canvas-engine/api';
import {
  ChevronRightIcon,
  EyeIcon,
  EyeOffIcon,
  FolderIcon,
  FolderOpenIcon,
  ImageIcon,
  LockIcon,
  LockOpenIcon,
} from 'lucide-react';
import { memo, useCallback, useLayoutEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerRowCommands } from './layerRowCommands';
import type { LayerTreeRow } from './layerTreeRows';

import { colorLabelHex } from './colorLabels';
import { ControlLayerWarningIcon } from './ControlLayerWarningIcon';
import { LayerActiveDot, ROW_SELECTION_FOCUS } from './LayerActiveDot';
import { recordLayerRowCommit } from './layerPanelDiagnostics';
import { LAYER_TREE_INDENT_PX } from './layerPanelRows';
import { anchorFromPoint } from './layerRowCommands';
import { layerRowSummary } from './layerRowSummary';
import { LayerRowSurface } from './LayerRowSurface';
import { LayerThumbnail, type LayerThumbnailEngine } from './LayerThumbnail';
import { layerTypeIcon } from './layerTypeIcon';

const CHEVRON_HOVER = { bg: 'bg.muted', color: 'fg' };

const THUMBNAIL_SIZE = '8';
const GROUP_PILE_OFFSET_PX = 4;
const GROUP_PILE_CARD_PX = 24;

/** How a row takes part in the current drag. */
export type LayerRowDragState = 'source' | 'travelling' | null;

interface LayerRowProps {
  /** Combined-set ARIA facts when the parent group owns modifier rows. */
  ariaPosInSet?: number;
  ariaSetSize?: number;
  /** A group on the primary layer's ancestor path; tinted as selection context. */
  ancestorOfPrimary: boolean;
  /** Projected child rows the layer owns; a chevron appears above zero. */
  childCount: number;
  /** One of this layer's child rows holds the sub-selection; the row yields the emphasis to it. */
  childSelected: boolean;
  childrenExpanded: boolean;
  commands: LayerRowCommands;
  drag: LayerRowDragState;
  /** Drag reordering is off: the editing lock or degraded mode. */
  dragDisabled: boolean;
  editingLocked: boolean;
  engine: LayerThumbnailEngine | null;
  focused: boolean;
  primary: boolean;
  renaming: boolean;
  row: LayerTreeRow;
  selected: boolean;
  /** Thumbnails are drawn; off in degraded mode. */
  thumbnails: boolean;
}

const stopPropagation = (event: { stopPropagation: () => void }): void => event.stopPropagation();

/**
 * One tree item. The row element itself carries the tree role, the roving tab stop and the
 * keyboard model; every control inside it is pointer-only (`tabIndex={-1}`) and reachable from the
 * keyboard through the row's menu, so the tree stays a single tab stop.
 */
const LayerRowComponent = ({
  ancestorOfPrimary,
  ariaPosInSet,
  ariaSetSize,
  childCount,
  childSelected,
  childrenExpanded,
  commands,
  drag,
  dragDisabled,
  editingLocked,
  engine,
  focused,
  primary,
  renaming,
  row,
  selected,
  thumbnails,
}: LayerRowProps) => {
  const { t } = useTranslation();
  const { vm } = row;
  const { node } = vm;
  const group = vm.kind === 'group';
  const layer = group ? null : (node as CanvasLayerContract);
  const { listeners, setNodeRef: setDragRef } = useDraggable({
    data: { stack: vm.stack },
    disabled: dragDisabled,
    id: row.id,
  });
  const { setNodeRef: setDropRef } = useDroppable({ data: { stack: vm.stack }, disabled: dragDisabled, id: row.id });
  const rowElement = useRef<HTMLDivElement | null>(null);
  const setRowRef = useCallback(
    (element: HTMLDivElement | null) => {
      rowElement.current = element;
      setDragRef(element);
      setDropRef(element);
    },
    [setDragRef, setDropRef]
  );
  const nameInput = useRef<HTMLInputElement | null>(null);
  // Escape abandons the draft; the blur that follows refocusing the row must not commit it.
  const renameCancelled = useRef(false);

  useLayoutEffect(() => {
    recordLayerRowCommit(row.id);
  });

  const indentStyle = useMemo(() => ({ paddingLeft: `${vm.depth * LAYER_TREE_INDENT_PX}px` }), [vm.depth]);

  const handleSelect = useCallback(
    (event: MouseEvent<HTMLElement>) =>
      commands.select(row.id, { additive: event.metaKey || event.ctrlKey, range: event.shiftKey }),
    [commands, row.id]
  );
  const handleFocus = useCallback(() => commands.focus(row.id), [commands, row.id]);
  const keepRowFocus = useCallback((event: MouseEvent<HTMLElement>) => {
    event.preventDefault();
    rowElement.current?.focus();
  }, []);
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLElement>) => {
      if (event.target !== event.currentTarget) {
        return;
      }
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        commands.select(row.id, { additive: event.metaKey || event.ctrlKey, range: event.shiftKey });
        return;
      }
      commands.keyDown(row.id, event);
    },
    [commands, row.id]
  );
  const handleToggleExpanded = useCallback(
    (event: { stopPropagation: () => void }) => {
      event.stopPropagation();
      commands.toggleExpanded(row.id);
    },
    [commands, row.id]
  );
  const handleToggleChildren = useCallback(
    (event: { stopPropagation: () => void }) => {
      event.stopPropagation();
      commands.toggleChildren(row.id);
    },
    [commands, row.id]
  );
  const handleToggleVisible = useCallback(
    (checked: boolean) => commands.setEnabled(row.id, checked),
    [commands, row.id]
  );
  const handleToggleHidden = useCallback(
    (event: { stopPropagation: () => void }) => {
      event.stopPropagation();
      commands.setHidden(row.id, !isNodeHidden(node));
    },
    [commands, node, row.id]
  );
  const handleToggleLock = useCallback(
    (event: { stopPropagation: () => void }) => {
      event.stopPropagation();
      commands.setLocked(row.id, !node.isLocked);
    },
    [commands, node.isLocked, row.id]
  );
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      event.preventDefault();
      if (!selected) {
        commands.select(row.id, { additive: false, range: false });
      }
      commands.openMenu(row.id, anchorFromPoint(event.clientX, event.clientY));
    },
    [commands, row.id, selected]
  );
  const startRename = useCallback(
    (event?: MouseEvent<HTMLElement>) => {
      if (event && (event.target as HTMLElement).closest('button, input')) {
        return;
      }
      if (!editingLocked) {
        commands.startRename(row.id);
      }
    },
    [commands, editingLocked, row.id]
  );
  const finishRename = useCallback(
    (refocus: boolean) => {
      commands.endRename();
      if (refocus) {
        rowElement.current?.focus();
      }
    },
    [commands]
  );
  const commitName = useCallback(
    (refocus: boolean) => {
      const name = nameInput.current?.value.trim() ?? '';
      const cancelled = renameCancelled.current;
      // Refocusing the row blurs the still-mounted input, which re-enters this
      // handler; claiming the commit first keeps it to one dispatch.
      renameCancelled.current = true;
      finishRename(refocus);
      if (!cancelled && name && name !== node.name) {
        commands.rename(row.id, name);
      }
    },
    [commands, finishRename, node.name, row.id]
  );
  // Restore dropped focus to the row, but explicitly preserve a destination chosen during input unmount.
  const handleNameBlur = useCallback(
    (event: FocusEvent<HTMLInputElement>) => {
      const next = event.relatedTarget;
      // The window losing focus is not a decision; the draft waits for it to come back.
      if (next === null && !window.document.hasFocus()) {
        return;
      }
      commitName(next === null);
      if (next instanceof HTMLElement) {
        next.focus();
      }
    },
    [commitName]
  );
  const handleNameKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      event.stopPropagation();
      if (event.key === 'Enter') {
        commitName(true);
      } else if (event.key === 'Escape') {
        renameCancelled.current = true;
        finishRename(true);
      }
    },
    [commitName, finishRename]
  );
  const focusOnMount = useCallback((input: HTMLInputElement | null) => {
    nameInput.current = input;
    renameCancelled.current = false;
    input?.focus();
    input?.select();
  }, []);

  const dragListeners = useMemo(() => {
    if (dragDisabled || !listeners) {
      return {};
    }
    const { onKeyDown: _onKeyDown, ...rest } = listeners;
    return rest;
  }, [dragDisabled, listeners]);

  const hideable = group ? isOverlayStack(vm.stack) : isHideableLayer(layer!);
  const ownHidden = isNodeHidden(node);
  const hiddenByAncestor = vm.documentHidden && !ownHidden;
  const lockedByAncestor = vm.effectiveLocked && !node.isLocked;
  const disabledByAncestor = !vm.contributionEnabled && node.isEnabled;
  const tone = primary
    ? childSelected
      ? 'muted'
      : 'emphasized'
    : selected
      ? 'selected'
      : ancestorOfPrimary
        ? 'muted'
        : undefined;

  return (
    <Box
      ref={setRowRef}
      {...dragListeners}
      aria-current={primary ? 'true' : undefined}
      aria-expanded={group ? row.expanded : childCount > 0 ? childrenExpanded : undefined}
      aria-label={node.name}
      aria-level={vm.depth + 2}
      aria-posinset={ariaPosInSet ?? row.posInSet}
      aria-selected={selected}
      aria-setsize={ariaSetSize ?? row.setSize}
      data-layer-row-id={row.id}
      data-primary={primary || undefined}
      h="full"
      opacity={drag ? 0.4 : undefined}
      pb="0.5"
      position="relative"
      role="treeitem"
      rounded="sm"
      tabIndex={focused ? 0 : -1}
      _focusVisible={ROW_SELECTION_FOCUS}
      onClick={handleSelect}
      onContextMenu={handleContextMenu}
      onDoubleClick={startRename}
      onFocus={handleFocus}
      onKeyDown={handleKeyDown}
    >
      {node.colorLabel ? (
        // Use a fixed, noninteractive swatch gutter at every nesting depth.
        <Box
          bg={colorLabelHex(node.colorLabel)}
          bottom="1.5"
          data-color-label={node.colorLabel}
          left="0"
          pointerEvents="none"
          position="absolute"
          rounded="full"
          top="1"
          w="2px"
        />
      ) : null}
      <LayerRowSurface active={tone} cursor={drag === 'source' ? 'grabbing' : 'default'} indentStyle={indentStyle}>
        <LayerActiveDot
          checked={node.isEnabled}
          disabled={editingLocked}
          gated={disabledByAncestor}
          label={t('widgets.layers.actions.toggleActive')}
          tooltip={disabledByAncestor ? t('widgets.layers.actions.groupDisabled') : undefined}
          onCheckedChange={handleToggleVisible}
          onKeepRowFocus={keepRowFocus}
        />
        {group ? (
          <GroupPreview engine={engine} expanded={row.expanded} node={node} thumbnails={thumbnails} />
        ) : !thumbnails ? (
          <Box
            alignItems="center"
            bg="bg.muted"
            borderColor="border.subtle"
            borderWidth="1px"
            boxSize={THUMBNAIL_SIZE}
            color="fg.muted"
            display="flex"
            flexShrink={0}
            justifyContent="center"
            rounded="sm"
          >
            <Icon as={ImageIcon} boxSize="4" />
          </Box>
        ) : (
          <Box boxSize={THUMBNAIL_SIZE} flexShrink={0}>
            <LayerThumbnail engine={engine} layer={layer!} />
          </Box>
        )}
        <Box
          alignItems="center"
          display="flex"
          flexDirection="column"
          flexShrink={0}
          h="6"
          justifyContent="center"
          w="5"
        >
          <Icon
            as={group ? (row.expanded ? FolderOpenIcon : FolderIcon) : layerTypeIcon(layer!)}
            boxSize="3"
            color="fg.subtle"
          />
          {group || childCount > 0 ? (
            <chakra.button
              aria-label={t(
                group
                  ? row.expanded
                    ? 'widgets.layers.actions.collapseGroup'
                    : 'widgets.layers.actions.expandGroup'
                  : childrenExpanded
                    ? 'widgets.layers.actions.hideModifiers'
                    : 'widgets.layers.actions.showModifiers'
              )}
              alignItems="center"
              color="fg.muted"
              display="flex"
              h="3"
              justifyContent="center"
              rounded="xs"
              tabIndex={-1}
              type="button"
              w="4"
              _hover={CHEVRON_HOVER}
              onClick={group ? handleToggleExpanded : handleToggleChildren}
              onMouseDown={keepRowFocus}
              onPointerDown={stopPropagation}
            >
              <Icon
                as={ChevronRightIcon}
                boxSize="3"
                transform={(group ? row.expanded : childrenExpanded) ? 'rotate(90deg)' : undefined}
                transitionDuration="fast"
                transitionProperty="transform"
              />
            </chakra.button>
          ) : null}
        </Box>
        <Box
          flex="1"
          minW="0"
          title={
            group
              ? vm.leafCount === 0
                ? t('widgets.layers.groupEmpty')
                : t('widgets.layers.groupSummary', { count: vm.leafCount })
              : layerRowSummary(layer!, t)
          }
        >
          {renaming ? (
            <Input
              ref={focusOnMount}
              aria-label={t('widgets.layers.actions.rename')}
              defaultValue={node.name}
              size="2xs"
              userSelect="text"
              onBlur={handleNameBlur}
              onClick={stopPropagation}
              onKeyDown={handleNameKeyDown}
              onPointerDown={stopPropagation}
            />
          ) : (
            <MiddleTruncate
              color={vm.contributionEnabled ? undefined : 'fg.muted'}
              fontSize="2xs"
              fontWeight="700"
              text={node.name}
            />
          )}
        </Box>
        {layer ? <ControlLayerWarningIcon contributing={vm.contributionEnabled} layer={layer} /> : null}
        {/* One control cluster on the same rhythm as the stack header; slots a row cannot use are held open. */}
        <HStack flexShrink="0" gap="0.5" onClick={stopPropagation} onMouseDown={keepRowFocus}>
          {hideable ? (
            <Tooltip
              content={
                hiddenByAncestor ? t('widgets.layers.actions.groupHidden') : t('widgets.layers.actions.toggleHidden')
              }
            >
              <IconButton
                aria-label={t('widgets.layers.actions.toggleHidden')}
                aria-pressed={!ownHidden}
                color={vm.documentHidden ? 'fg.muted' : 'fg'}
                disabled={editingLocked || hiddenByAncestor}
                size="2xs"
                tabIndex={-1}
                variant="ghost"
                onClick={handleToggleHidden}
                onPointerDown={stopPropagation}
              >
                {vm.documentHidden ? <EyeOffIcon /> : <EyeIcon />}
              </IconButton>
            </Tooltip>
          ) : (
            <Box boxSize="6" />
          )}
          <Tooltip
            content={
              lockedByAncestor ? t('widgets.layers.actions.groupLocked') : t('widgets.layers.actions.toggleLock')
            }
          >
            <IconButton
              aria-label={t('widgets.layers.actions.toggleLock')}
              color={node.isLocked ? 'fg' : 'fg.muted'}
              disabled={editingLocked || lockedByAncestor}
              size="2xs"
              tabIndex={-1}
              variant="ghost"
              onClick={handleToggleLock}
              onPointerDown={stopPropagation}
            >
              {vm.effectiveLocked ? <LockIcon /> : <LockOpenIcon />}
            </IconButton>
          </Tooltip>
        </HStack>
      </LayerRowSurface>
    </Box>
  );
};

export const LayerRow = memo(LayerRowComponent);

/** The first few leaf layers of a group, depth first — what its preview pile shows. */
const groupLeafPreviews = (node: CanvasNodeContract, limit = 3): CanvasLayerContract[] => {
  const leaves: CanvasLayerContract[] = [];
  const walk = (candidate: CanvasNodeContract): void => {
    if (leaves.length >= limit) {
      return;
    }
    if (isGroupNode(candidate)) {
      for (const child of candidate.children) {
        walk(child);
      }
      return;
    }
    leaves.push(candidate);
  };
  if (isGroupNode(node)) {
    for (const child of node.children) {
      walk(child);
    }
  }
  return leaves;
};

const GroupPreview = ({
  engine,
  expanded,
  node,
  thumbnails,
}: {
  engine: LayerThumbnailEngine | null;
  expanded: boolean;
  node: CanvasNodeContract;
  thumbnails: boolean;
}) => {
  const leaves = useMemo(() => (thumbnails ? groupLeafPreviews(node) : []), [node, thumbnails]);
  const cardStyles = useMemo(
    () =>
      leaves.map((_, index): CSSProperties => {
        const offset = index * GROUP_PILE_OFFSET_PX;
        return { height: GROUP_PILE_CARD_PX, left: offset, top: offset, width: GROUP_PILE_CARD_PX };
      }),
    [leaves]
  );
  if (leaves.length === 0) {
    return (
      <Box
        alignItems="center"
        bg="bg.muted"
        borderColor="border.subtle"
        borderWidth="1px"
        boxSize={THUMBNAIL_SIZE}
        color="fg.muted"
        display="flex"
        flexShrink={0}
        justifyContent="center"
        rounded="sm"
      >
        <Icon as={expanded ? FolderOpenIcon : FolderIcon} boxSize="4" />
      </Box>
    );
  }
  return (
    <Box boxSize={THUMBNAIL_SIZE} flexShrink={0} position="relative">
      {leaves
        .map((leaf, index) => (
          <Box key={leaf.id} overflow="hidden" position="absolute" rounded="sm" style={cardStyles[index]}>
            <LayerThumbnail engine={engine} layer={leaf} />
          </Box>
        ))
        .reverse()}
    </Box>
  );
};

/** The compact card that follows the pointer: the grabbed row's name plus how many rows travel. */
export const LayerDragGhost = ({ count, vm }: { count: number; vm: SemanticNode }) => (
  <HStack
    bg="bg.panel"
    borderColor="accent.solid"
    borderWidth="1px"
    boxShadow="lg"
    cursor="grabbing"
    gap="2"
    maxW="16rem"
    px="2"
    py="1.5"
    rounded="sm"
  >
    {vm.kind === 'group' ? <Icon as={FolderIcon} boxSize="3.5" color="fg.muted" flexShrink={0} /> : null}
    <Text flex="1" fontSize="2xs" fontWeight="700" truncate>
      {vm.node.name}
    </Text>
    {count > 1 ? (
      <Badge colorPalette="accent" size="xs" variant="solid">
        {count}
      </Badge>
    ) : null}
  </HStack>
);

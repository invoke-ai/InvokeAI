import type { CSSProperties, FocusEvent, KeyboardEvent, MouseEvent } from 'react';

import { Box, HStack, Icon, Input, Text } from '@chakra-ui/react';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import {
  ApertureIcon,
  WandSparklesIcon,
  ContrastIcon,
  DropletIcon,
  GaugeIcon,
  ImageIcon,
  RainbowIcon,
  SlidersVerticalIcon,
  SplineIcon,
  SunMediumIcon,
  WavesIcon,
  type LucideIcon,
} from 'lucide-react';
import { memo, useCallback, useLayoutEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerRowCommands } from './layerRowCommands';

import { LayerActiveDot, ROW_SELECTION_FOCUS } from './LayerActiveDot';
import { childRowNameKey, isOrderedChildKind, type LayerChildRowKind, type ProjectedChildRow } from './layerChildRows';
import { recordLayerRowCommit } from './layerPanelDiagnostics';
import { LAYER_TREE_INDENT_PX } from './layerPanelRows';
import { anchorFromPoint } from './layerRowCommands';
import { LayerRowSurface } from './LayerRowSurface';

const THUMBNAIL_IMG_STYLE: CSSProperties = { height: '100%', objectFit: 'cover', width: '100%' };

const stopPropagation = (event: { stopPropagation: () => void }): void => event.stopPropagation();

const CHILD_ROW_GLYPHS: Record<LayerChildRowKind, LucideIcon> = {
  'adjustment-brightness-contrast': SunMediumIcon,
  'adjustment-curves': SplineIcon,
  'adjustment-exposure': ApertureIcon,
  'adjustment-hsl': DropletIcon,
  'adjustment-hue': RainbowIcon,
  'adjustment-invert': ContrastIcon,
  'adjustment-levels': SlidersVerticalIcon,
  'layer-region': WandSparklesIcon,
  'mask-denoise': GaugeIcon,
  'mask-noise': WavesIcon,
  'reference-image': ImageIcon,
};

/** Kinds whose rows can be picked up: reorderable entries and movable reference images. */
export const isDraggableChildKind = (kind: LayerChildRowKind): boolean =>
  kind === 'reference-image' || isOrderedChildKind(kind);

/** Kinds whose rows carry a user-given name: adjustment entries. */
export const isRenameableChildKind = (kind: LayerChildRowKind): boolean =>
  kind === 'layer-region' || isOrderedChildKind(kind);

/** The row's display name: its custom name, a reference image's number, or its kind's name. */
export const childRowName = (child: ProjectedChildRow, t: (key: string) => string): string =>
  child.kind === 'reference-image'
    ? `${t('widgets.layers.regionalGuidance.referenceImage')} ${child.posInSet}`
    : (child.customName ?? t(childRowNameKey(child.kind)));

interface LayerChildRowProps {
  /** The owner's combined child count when a group owns both rows and nodes. */
  ariaSetSize?: number;
  child: ProjectedChildRow;
  commands: LayerRowCommands;
  /** The owning layer travels in the current drag; the row dims with it. */
  dimmed: boolean;
  dragDisabled: boolean;
  editingLocked: boolean;
  focused: boolean;
  renaming: boolean;
  selected: boolean;
}

/**
 * Modifier child rows share tree focus and route Properties; toggle/Delete edit them, while drops land below their
 * owning layer.
 */
const LayerChildRowComponent = ({
  ariaSetSize,
  child,
  commands,
  dimmed,
  dragDisabled,
  editingLocked,
  focused,
  renaming,
  selected,
}: LayerChildRowProps) => {
  const { t } = useTranslation();
  const rowElement = useRef<HTMLDivElement | null>(null);
  const nameInput = useRef<HTMLInputElement | null>(null);
  const renameCancelled = useRef(false);

  useLayoutEffect(() => {
    recordLayerRowCommit(child.key);
  });
  const { setNodeRef: setDropRef } = useDroppable({
    data: { stack: child.stack },
    disabled: dragDisabled,
    id: child.key,
  });
  const { listeners, setNodeRef: setDragRef } = useDraggable({
    data: { stack: child.stack },
    disabled: dragDisabled || !isDraggableChildKind(child.kind),
    id: child.key,
  });
  const setRowRef = useCallback(
    (element: HTMLDivElement | null) => {
      rowElement.current = element;
      setDropRef(element);
      setDragRef(element);
    },
    [setDragRef, setDropRef]
  );
  const dragListeners = useMemo(() => {
    if (dragDisabled || !listeners) {
      return {};
    }
    const { onKeyDown: _onKeyDown, ...rest } = listeners;
    return rest;
  }, [dragDisabled, listeners]);
  const name = childRowName(child, t);

  const indentStyle = useMemo(() => ({ paddingLeft: `${child.depth * LAYER_TREE_INDENT_PX}px` }), [child.depth]);
  const handleSelect = useCallback(() => commands.selectChild(child), [child, commands]);
  const handleFocus = useCallback(() => commands.focus(child.key), [child.key, commands]);
  const keepRowFocus = useCallback((event: MouseEvent<HTMLElement>) => {
    event.preventDefault();
    rowElement.current?.focus();
  }, []);
  const handleToggle = useCallback((checked: boolean) => commands.setChildEnabled(child, checked), [child, commands]);
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLElement>) => {
      if (event.target !== event.currentTarget) {
        return;
      }
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        commands.selectChild(child);
        return;
      }
      commands.keyDown(child.key, event);
    },
    [child, commands]
  );
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      event.preventDefault();
      if (!selected) {
        commands.selectChild(child, { reveal: false });
      }
      commands.openChildMenu(child, anchorFromPoint(event.clientX, event.clientY));
    },
    [child, commands, selected]
  );
  const startRename = useCallback(
    (event?: MouseEvent<HTMLElement>) => {
      if (event && (event.target as HTMLElement).closest('button, input')) {
        return;
      }
      if (!editingLocked && isRenameableChildKind(child.kind)) {
        commands.startRename(child.key);
      }
    },
    [child.key, child.kind, commands, editingLocked]
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
  const defaultName = child.kind === 'reference-image' ? null : t(childRowNameKey(child.kind));
  const commitName = useCallback(
    (refocus: boolean) => {
      const draft = nameInput.current?.value.trim() ?? '';
      const cancelled = renameCancelled.current;
      // Refocusing the row blurs the still-mounted input, which re-enters this
      // handler; claiming the commit first keeps it to one dispatch.
      renameCancelled.current = true;
      finishRename(refocus);
      if (!cancelled) {
        commands.renameChild(child, draft.length > 0 && draft !== defaultName ? draft : null);
      }
    },
    [child, commands, defaultName, finishRename]
  );
  const handleNameBlur = useCallback(
    (event: FocusEvent<HTMLInputElement>) => {
      const next = event.relatedTarget;
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

  const muted = !child.isEnabled || !child.parentContributing;

  return (
    <Box
      ref={setRowRef}
      {...dragListeners}
      aria-label={name}
      aria-level={child.depth + 2}
      aria-posinset={child.posInSet}
      aria-selected={selected}
      aria-setsize={ariaSetSize ?? child.setSize}
      data-layer-row-id={child.key}
      h="full"
      opacity={dimmed ? 0.4 : undefined}
      pb="0.5"
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
      <LayerRowSurface active={selected ? 'emphasized' : undefined} indentStyle={indentStyle}>
        <LayerActiveDot
          checked={child.isEnabled}
          disabled={editingLocked}
          gated={!child.parentContributing}
          label={t('widgets.layers.modifiers.toggleActive')}
          tooltip={child.parentContributing ? undefined : t('widgets.layers.modifiers.parentDisabled')}
          onCheckedChange={handleToggle}
          onKeepRowFocus={keepRowFocus}
        />
        <Box
          alignItems="center"
          bg="bg.muted"
          borderColor="border.subtle"
          borderWidth="1px"
          boxSize="6"
          color="fg.muted"
          display="flex"
          flexShrink={0}
          justifyContent="center"
          overflow="hidden"
          rounded="sm"
        >
          {child.image ? (
            <img alt="" draggable={false} src={child.image.thumbnailUrl} style={THUMBNAIL_IMG_STYLE} />
          ) : (
            <Icon as={CHILD_ROW_GLYPHS[child.kind]} boxSize="3" />
          )}
        </Box>
        {renaming ? (
          <Input
            ref={focusOnMount}
            aria-label={t('widgets.layers.actions.rename')}
            defaultValue={name}
            flex="1"
            minW="0"
            size="2xs"
            userSelect="text"
            onBlur={handleNameBlur}
            onClick={stopPropagation}
            onKeyDown={handleNameKeyDown}
            onPointerDown={stopPropagation}
          />
        ) : (
          <Text color={muted ? 'fg.muted' : undefined} flex="1" fontSize="2xs" fontWeight="600" minW="0" truncate>
            {name}
          </Text>
        )}
        {child.detail !== null && !renaming ? (
          <Text color="fg.subtle" flexShrink={0} fontSize="2xs" fontVariantNumeric="tabular-nums">
            {child.detail}
          </Text>
        ) : null}
      </LayerRowSurface>
    </Box>
  );
};

export const LayerChildRow = memo(LayerChildRowComponent);

/** The compact card that follows the pointer while a child row is dragged. */
export const ChildDragGhost = ({ child }: { child: ProjectedChildRow }) => {
  const { t } = useTranslation();
  return (
    <HStack
      bg="bg.panel"
      borderColor="accent.solid"
      borderWidth="1px"
      boxShadow="lg"
      cursor="grabbing"
      gap="2"
      maxW="14rem"
      px="2"
      py="1"
      rounded="sm"
    >
      <Icon as={CHILD_ROW_GLYPHS[child.kind]} boxSize="3" color="fg.muted" flexShrink={0} />
      <Text fontSize="2xs" fontWeight="700" truncate>
        {childRowName(child, t)}
      </Text>
    </HStack>
  );
};

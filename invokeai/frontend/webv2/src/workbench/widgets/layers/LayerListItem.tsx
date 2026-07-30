import type { CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { CanvasEngineHandle } from '@workbench/widgets/canvas/useCanvasEngine';
import type { Dispatch, KeyboardEvent, MouseEvent } from 'react';

import { Badge, Box, chakra, HStack, Input, Stack, Text } from '@chakra-ui/react';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { IconButton, Row, ToggleDot } from '@platform/ui';
import { isHideableLayer, isLayerHidden } from '@workbench/canvas-engine/api';
import { EyeIcon, EyeOffIcon, GripVerticalIcon, LockIcon, LockOpenIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { ControlLayerWarningIcon } from './ControlLayerWarningIcon';
import {
  CanvasLayerContextMenu,
  type CanvasLayerContextMenuTarget,
  LayerContextMenu,
  type LayerContextMenuEngine,
} from './LayerContextMenu';
import { shouldStartLayerKeyboardDrag } from './layerDndConfig';
import { createLayerMenuTargetFromContextEvent } from './layerMenuState';
import { applyStructural } from './layerOps';
import { LayerPropertiesPopover, type LayerPropertiesEngine } from './LayerPropertiesPopover';
import { LayerThumbnail } from './LayerThumbnail';

const ROW_INTERACTIVE_DESCENDANTS = {
  '& button, & input': {
    pointerEvents: 'auto',
  },
};
const ROW_SELECTION_FOCUS = {
  outline: '2px solid',
  outlineColor: 'accent.solid',
  outlineOffset: '-2px',
};
const VISIBILITY_DOT_BASE = {
  borderRadius: 'full',
  borderWidth: '1px',
  content: '""',
  h: '3',
  inset: '50% auto auto 50%',
  position: 'absolute',
  transform: 'translate(-50%, -50%)',
  transition: 'background var(--wb-motion-duration-fast), border-color var(--wb-motion-duration-fast)',
  w: '3',
};
const VISIBILITY_DOT_CHECKED = {
  ...VISIBILITY_DOT_BASE,
  bg: 'accent.solid',
  borderColor: 'accent.solid',
};
const VISIBILITY_DOT_UNCHECKED = {
  ...VISIBILITY_DOT_BASE,
  bg: 'transparent',
  borderColor: 'border.emphasized',
};
const VISIBILITY_DOT_CHECKED_HOVER = {
  _before: {
    bg: 'accent.emphasized',
    borderColor: 'accent.emphasized',
  },
};
const VISIBILITY_DOT_UNCHECKED_HOVER = {
  _before: {
    borderColor: 'fg.muted',
  },
};

export type LayerListItemEngine = LayerContextMenuEngine & LayerPropertiesEngine & Pick<CanvasEngineHandle, 'previews'>;

/** i18n key for a layer's short type/source badge. */
const layerBadgeKey = (layer: CanvasLayerContract): string => {
  if (layer.type === 'raster') {
    return layer.source.type === 'image' ? 'widgets.layers.types.image' : 'widgets.layers.types.paint';
  }
  return `widgets.layers.types.${layer.type}`;
};

interface LayerListItemProps {
  dispatch: Dispatch<CanvasProjectMutation>;
  editingLocked: boolean;
  engine: LayerListItemEngine | null;
  index: number;
  isSelected: boolean;
  layer: CanvasLayerContract;
  layers: readonly CanvasLayerContract[];
}

export const getLayerListItemInteractionState = (editingLocked: boolean) => ({
  canRename: !editingLocked,
  canSelect: true,
  canToggleLock: !editingLocked,
  canToggleVisibility: !editingLocked,
  sortableDisabled: editingLocked,
});

/**
 * One layer row: thumbnail, name (double-click to rename), type badge,
 * visibility + lock toggles, a properties popover (blend mode + the layer's
 * type-specific settings), and an overflow/context menu. The whole row remains
 * the pointer drag target, while a dedicated reorder button owns keyboard
 * sorting. The pointer distance constraint keeps clicks, double-click rename,
 * and row buttons working; selected-layer opacity lives in the panel header.
 */
export const LayerListItem = ({
  dispatch,
  editingLocked,
  engine,
  index,
  isSelected,
  layer,
  layers,
}: LayerListItemProps) => {
  const { t } = useTranslation();
  const interaction = getLayerListItemInteractionState(editingLocked);
  const { attributes, isDragging, listeners, setActivatorNodeRef, setNodeRef, transform, transition } = useSortable({
    disabled: interaction.sortableDisabled,
    id: layer.id,
  });
  const rowRef = useRef<HTMLElement | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [draftName, setDraftName] = useState(layer.name);
  const [contextMenuTarget, setContextMenuTarget] = useState<CanvasLayerContextMenuTarget | null>(null);

  const dndStyle = useMemo(
    () => ({
      opacity: isDragging ? 0.4 : undefined,
      position: 'relative' as const,
      transform: CSS.Translate.toString(transform),
      transition,
      zIndex: isDragging ? 1 : undefined,
    }),
    [isDragging, transform, transition]
  );

  const handleSelect = useCallback(() => {
    if (interaction.canSelect && !isSelected) {
      dispatch({ id: layer.id, type: 'setCanvasSelectedLayer' });
    }
  }, [dispatch, interaction.canSelect, isSelected, layer.id]);

  const patchBase = useCallback(
    (label: string, forward: Partial<CanvasLayerContract>, inverse: Partial<CanvasLayerContract>) => {
      applyStructural(
        engine,
        dispatch,
        label,
        { id: layer.id, patch: forward, type: 'updateCanvasLayer' },
        { id: layer.id, patch: inverse, type: 'updateCanvasLayer' }
      );
    },
    [dispatch, engine, layer.id]
  );

  const handleToggleVisible = useCallback(
    (checked: boolean) => {
      patchBase(t('widgets.layers.actions.toggleVisibility'), { isEnabled: checked }, { isEnabled: layer.isEnabled });
    },
    [layer.isEnabled, patchBase, t]
  );

  /**
   * Hide is a DISPLAY-only axis, orthogonal to enabled: a hidden control map or
   * mask still conditions generation exactly as it would if visible. Only the
   * three overlay types have it — for a raster layer, visibility and
   * participation are the same fact.
   */
  const handleToggleHidden = useCallback(
    (event: { stopPropagation: () => void }) => {
      event.stopPropagation();
      const isHidden = !isLayerHidden(layer);
      applyStructural(
        engine,
        dispatch,
        t('widgets.layers.actions.toggleHidden'),
        { type: 'setCanvasLayersHidden', updates: [{ id: layer.id, isHidden }] },
        { type: 'setCanvasLayersHidden', updates: [{ id: layer.id, isHidden: !isHidden }] }
      );
    },
    [dispatch, engine, layer, t]
  );

  const handleToggleLock = useCallback(
    (event: { stopPropagation: () => void }) => {
      event.stopPropagation();
      patchBase(t('widgets.layers.actions.toggleLock'), { isLocked: !layer.isLocked }, { isLocked: layer.isLocked });
    },
    [layer.isLocked, patchBase, t]
  );

  const startEditing = useCallback(() => {
    setDraftName(layer.name);
    setIsEditing(true);
  }, [layer.name]);

  const commitName = useCallback(() => {
    setIsEditing(false);
    const name = draftName.trim();
    if (name && name !== layer.name) {
      patchBase(t('widgets.layers.actions.rename'), { name }, { name: layer.name });
    }
  }, [draftName, layer.name, patchBase, t]);

  const handleNameKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      // Stop ONLY the two keys the rename owns. A blanket stop would also stop
      // the native event reaching the window, killing every global hotkey while
      // renaming; the hotkey runtime already refuses to fire non-`allowInEditable`
      // bindings for a focused input, so nothing else needs suppressing here.
      // Escape does need it: the engine's own window listener would otherwise
      // take it as a canvas deselect.
      if (event.key === 'Enter') {
        event.stopPropagation();
        commitName();
      } else if (event.key === 'Escape') {
        event.stopPropagation();
        setIsEditing(false);
      }
    },
    [commitName]
  );

  const handleNameChange = useCallback((event: { target: { value: string } }) => setDraftName(event.target.value), []);

  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      if (!isSelected) {
        dispatch({ id: layer.id, type: 'setCanvasSelectedLayer' });
      }
      setContextMenuTarget(createLayerMenuTargetFromContextEvent(layer.id, event));
    },
    [dispatch, isSelected, layer.id]
  );

  const closeContextMenu = useCallback(() => setContextMenuTarget(null), []);

  // The row's own DOM node, so the drag activator can tell a keystroke made
  // inside it from one that bubbled in through the React tree from a portal.
  const setRowRef = useCallback(
    (node: HTMLElement | null) => {
      rowRef.current = node;
      setNodeRef(node);
    },
    [setNodeRef]
  );

  /**
   * dnd-kit starts a drag on Enter and inspects only the key code, so the
   * activator is gated here — see {@link shouldStartLayerKeyboardDrag}. Without
   * it, `mod+Enter` (Invoke) with a row focused starts an invisible drag that
   * leaves the row at drag opacity, reading as a disabled layer.
   */
  const sortableListeners = useMemo(() => {
    if (interaction.sortableDisabled || !listeners) {
      return { handle: {}, row: {} };
    }
    const { onKeyDown, ...rest } = listeners;
    return {
      handle: {
        onKeyDown: (event: KeyboardEvent<HTMLElement>) => {
          if (!shouldStartLayerKeyboardDrag(event, rowRef.current)) {
            return;
          }
          onKeyDown?.(event);
        },
      },
      row: rest,
    };
  }, [interaction.sortableDisabled, listeners]);

  return (
    <Box ref={setRowRef} style={dndStyle}>
      <Row
        {...sortableListeners.row}
        active={isSelected ? 'muted' : undefined}
        cursor={isDragging ? 'grabbing' : 'default'}
        display="flex"
        gap="1.5"
        p="1.5"
        position="relative"
        onContextMenu={handleContextMenu}
      >
        <chakra.button
          aria-label={t('widgets.layers.actions.select', { name: layer.name })}
          aria-pressed={isSelected}
          inset="0"
          position="absolute"
          rounded="sm"
          type="button"
          _focusVisible={ROW_SELECTION_FOCUS}
          onClick={handleSelect}
          onDoubleClick={interaction.canRename ? startEditing : undefined}
        />
        <HStack css={ROW_INTERACTIVE_DESCENDANTS} gap="1.5" pointerEvents="none" position="relative" w="full">
          <IconButton
            ref={setActivatorNodeRef}
            {...(interaction.sortableDisabled ? {} : attributes)}
            {...sortableListeners.handle}
            aria-label={`${t('widgets.layers.actions.reorder')}: ${layer.name}`}
            color="fg.subtle"
            cursor={isDragging ? 'grabbing' : 'grab'}
            disabled={interaction.sortableDisabled}
            size="2xs"
            variant="ghost"
          >
            <GripVerticalIcon aria-hidden="true" />
          </IconButton>
          <LayerThumbnail engine={engine} layer={layer} />
          <Stack flex="1" gap="0.5" minW="0">
            {isEditing ? (
              <Input
                autoFocus
                aria-label={t('widgets.layers.actions.rename')}
                disabled={!interaction.canRename}
                size="2xs"
                value={draftName}
                onBlur={commitName}
                onChange={handleNameChange}
                onKeyDown={handleNameKeyDown}
                onPointerDown={stopPropagation}
              />
            ) : (
              <Text aria-disabled={!interaction.canRename} fontSize="2xs" fontWeight="700" truncate>
                {layer.name}
              </Text>
            )}
            <HStack alignSelf="flex-start" gap="1">
              <Badge colorPalette="gray" size="xs" variant="subtle">
                {t(layerBadgeKey(layer))}
              </Badge>
              <ControlLayerWarningIcon layer={layer} />
            </HStack>
          </Stack>
          {isHideableLayer(layer) ? (
            <IconButton
              aria-label={t('widgets.layers.actions.toggleHidden')}
              aria-pressed={!isLayerHidden(layer)}
              color={isLayerHidden(layer) ? 'fg.subtle' : 'fg'}
              disabled={!interaction.canToggleVisibility}
              size="2xs"
              variant="ghost"
              onClick={handleToggleHidden}
              onPointerDown={stopPropagation}
            >
              {isLayerHidden(layer) ? <EyeOffIcon /> : <EyeIcon />}
            </IconButton>
          ) : null}
          <Box flexShrink="0" onClick={stopPropagation} onPointerDown={stopPropagation}>
            <ToggleDot
              _before={layer.isEnabled ? VISIBILITY_DOT_CHECKED : VISIBILITY_DOT_UNCHECKED}
              _focusVisible={ROW_SELECTION_FOCUS}
              _hover={layer.isEnabled ? VISIBILITY_DOT_CHECKED_HOVER : VISIBILITY_DOT_UNCHECKED_HOVER}
              bg="transparent"
              borderWidth="0"
              checked={layer.isEnabled}
              cursor={interaction.canToggleVisibility ? 'pointer' : 'not-allowed'}
              disabled={!interaction.canToggleVisibility}
              h="6"
              label={t('widgets.layers.actions.toggleVisibility')}
              position="relative"
              transition="none"
              w="6"
              onCheckedChange={handleToggleVisible}
            />
          </Box>
          <IconButton
            aria-label={t('widgets.layers.actions.toggleLock')}
            color={layer.isLocked ? 'fg' : 'fg.subtle'}
            disabled={!interaction.canToggleLock}
            size="2xs"
            variant="ghost"
            onClick={handleToggleLock}
            onPointerDown={stopPropagation}
          >
            {layer.isLocked ? <LockIcon /> : <LockOpenIcon />}
          </IconButton>
          <Box flexShrink="0" onClick={stopPropagation} onPointerDown={stopPropagation}>
            <LayerPropertiesPopover dispatch={dispatch} engine={engine} layer={layer} />
          </Box>
          <Box flexShrink="0" onPointerDown={stopPropagation}>
            <LayerContextMenu dispatch={dispatch} engine={engine} index={index} layer={layer} layers={layers} />
          </Box>
        </HStack>
      </Row>
      <CanvasLayerContextMenu
        dispatch={dispatch}
        engine={engine}
        layers={layers}
        target={contextMenuTarget}
        onClose={closeContextMenu}
      />
    </Box>
  );
};

const stopPropagation = (event: { stopPropagation: () => void }): void => event.stopPropagation();

import type { CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { CanvasEngineHandle } from '@workbench/widgets/canvas/useCanvasEngine';
import type { Dispatch, KeyboardEvent, MouseEvent } from 'react';

import { Badge, Box, chakra, HStack, Input, Stack } from '@chakra-ui/react';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { IconButton, Row, ToggleDot } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { isHideableLayer, isLayerHidden } from '@workbench/canvas-engine/api';
import { EyeIcon, EyeOffIcon, LockIcon, LockOpenIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { ControlLayerWarningIcon } from './ControlLayerWarningIcon';
import {
  CanvasLayerContextMenu,
  type CanvasLayerContextMenuTarget,
  LayerContextMenu,
  type LayerContextMenuEngine,
} from './LayerContextMenu';
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

  /**
   * Pointer listeners only. dnd-kit starts a keyboard drag on Enter and
   * inspects nothing but the key code, so with no grip to own that gesture the
   * activator is left unattached — Enter on a row belongs to selection, and
   * `mod+Enter` (Invoke) can no longer begin an invisible drag that leaves the
   * row at drag opacity. Keyboard reordering is the context menu's Move actions.
   */
  const sortableRowListeners = useMemo(() => {
    if (interaction.sortableDisabled || !listeners) {
      return {};
    }
    const { onKeyDown: _onKeyDown, ...rest } = listeners;
    return rest;
  }, [interaction.sortableDisabled, listeners]);

  // The selection button carries them, and owns its own role, focus, and
  // pressed state — dnd-kit's versions of those would overwrite it.
  const sortableAttributes = useMemo(() => {
    if (interaction.sortableDisabled) {
      return {};
    }
    const { role: _role, tabIndex: _tabIndex, 'aria-pressed': _ariaPressed, ...rest } = attributes;
    return rest;
  }, [attributes, interaction.sortableDisabled]);

  return (
    <Box ref={setNodeRef} style={dndStyle}>
      <Row
        {...sortableRowListeners}
        active={isSelected ? 'muted' : undefined}
        cursor={isDragging ? 'grabbing' : 'default'}
        display="flex"
        gap="1.5"
        p="1.5"
        position="relative"
        onContextMenu={handleContextMenu}
      >
        {/*
          No grip: the row itself is the pointer drag target, and the
          row-covering selection button carries the sortable role description so
          the affordance is still announced. dnd-kit's `role` / `tabIndex` /
          `aria-pressed` are dropped so the button keeps its own selection
          semantics — notably Enter, which selects rather than starting a drag.
          Keyboard reordering is the context menu's Move actions.
        */}
        <chakra.button
          ref={setActivatorNodeRef}
          {...sortableAttributes}
          aria-label={t('widgets.layers.actions.select', { name: layer.name })}
          aria-pressed={isSelected}
          cursor={isDragging ? 'grabbing' : undefined}
          inset="0"
          position="absolute"
          rounded="sm"
          type="button"
          _focusVisible={ROW_SELECTION_FOCUS}
          onClick={handleSelect}
          onDoubleClick={interaction.canRename ? startEditing : undefined}
        />
        <HStack css={ROW_INTERACTIVE_DESCENDANTS} gap="1.5" pointerEvents="none" position="relative" w="full">
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
              <MiddleTruncate
                aria-disabled={!interaction.canRename}
                fontSize="2xs"
                fontWeight="700"
                text={layer.name}
              />
            )}
            <HStack alignSelf="flex-start" gap="1">
              <Badge colorPalette="gray" size="xs" variant="subtle">
                {t(layerBadgeKey(layer))}
              </Badge>
              <ControlLayerWarningIcon layer={layer} />
            </HStack>
          </Stack>
          {/*
            One control cluster on the same 0.5 rhythm as the group header's, so
            every row's trailing icons land in the same columns as each other and
            as the header above. Raster layers have no display axis, so their
            hide slot is held open rather than collapsed — otherwise their
            remaining controls would shift out of column against masks.
          */}
          <HStack flexShrink="0" gap="0.5">
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
            ) : (
              <Box boxSize="6" />
            )}
            {/*
              `display="flex"` is load bearing, not cosmetic: as a block box this
              wrapper lays its button out on a text baseline, and the descender
              space below made it 30px tall and pushed the control 3px above the
              centre line its sibling icon buttons sit on.
            */}
            <Box display="flex" flexShrink="0" onClick={stopPropagation} onPointerDown={stopPropagation}>
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
            <Box display="flex" flexShrink="0" onClick={stopPropagation} onPointerDown={stopPropagation}>
              <LayerPropertiesPopover dispatch={dispatch} engine={engine} layer={layer} />
            </Box>
            <Box display="flex" flexShrink="0" onPointerDown={stopPropagation}>
              <LayerContextMenu dispatch={dispatch} engine={engine} index={index} layer={layer} layers={layers} />
            </Box>
          </HStack>
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

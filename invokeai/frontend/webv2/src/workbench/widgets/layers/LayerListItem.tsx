import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { CanvasLayerContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { Dispatch, KeyboardEvent, MouseEvent } from 'react';

import { Badge, Box, HStack, Input, Stack, Text } from '@chakra-ui/react';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { IconButton, Row, ToggleDot } from '@workbench/components/ui';
import { LockIcon, LockOpenIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { CanvasLayerContextMenu, type CanvasLayerContextMenuTarget, LayerContextMenu } from './LayerContextMenu';
import { createLayerMenuTargetFromContextEvent } from './layerMenuState';
import { applyStructural } from './layerOps';
import { LayerPropertiesPopover } from './LayerPropertiesPopover';
import { LayerThumbnail } from './LayerThumbnail';

/** i18n key for a layer's short type/source badge. */
const layerBadgeKey = (layer: CanvasLayerContract): string => {
  if (layer.type === 'raster') {
    return layer.source.type === 'image' ? 'widgets.layers.types.image' : 'widgets.layers.types.paint';
  }
  return `widgets.layers.types.${layer.type}`;
};

interface LayerListItemProps {
  dispatch: Dispatch<WorkbenchAction>;
  engine: CanvasEngine | null;
  index: number;
  isSelected: boolean;
  layer: CanvasLayerContract;
  layers: readonly CanvasLayerContract[];
}

/**
 * One layer row: thumbnail, name (double-click to rename), type badge,
 * visibility + lock toggles, a properties popover (blend mode + the layer's
 * type-specific settings), and an overflow/context menu. The whole row is the
 * drag target — no visible handle — with a pointer distance activation
 * constraint so clicks, double-click rename, and the row buttons still work;
 * the selected layer's opacity (and mask fill swatch) lives in the panel header.
 */
export const LayerListItem = ({ dispatch, engine, index, isSelected, layer, layers }: LayerListItemProps) => {
  const { t } = useTranslation();
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({ id: layer.id });
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
    if (!isSelected) {
      dispatch({ id: layer.id, type: 'setCanvasSelectedLayer' });
    }
  }, [dispatch, isSelected, layer.id]);

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
      // Keep every keystroke inside the rename input. The whole row carries
      // dnd-kit keyboard `listeners` whose activator claims Space/Enter (and
      // preventDefaults them) to arm a keyboard drag — without this guard, typing
      // a space in a layer name is eaten and can start a drag. Stop propagation
      // for ALL keys (not just the two we handle) so nothing bubbles to the row.
      event.stopPropagation();
      if (event.key === 'Enter') {
        commitName();
      } else if (event.key === 'Escape') {
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

  return (
    <Box ref={setNodeRef} style={dndStyle}>
      <Row
        {...attributes}
        {...listeners}
        active={isSelected ? 'muted' : undefined}
        cursor={isDragging ? 'grabbing' : 'default'}
        display="flex"
        gap="1.5"
        p="1.5"
        onClick={handleSelect}
        onContextMenu={handleContextMenu}
      >
        <HStack gap="1.5" w="full">
          <LayerThumbnail engine={engine} layer={layer} />
          <Stack flex="1" gap="0.5" minW="0">
            {isEditing ? (
              <Input
                autoFocus
                aria-label={t('widgets.layers.actions.rename')}
                size="2xs"
                value={draftName}
                onBlur={commitName}
                onChange={handleNameChange}
                onKeyDown={handleNameKeyDown}
                onPointerDown={stopPropagation}
              />
            ) : (
              <Text fontSize="2xs" fontWeight="700" truncate onDoubleClick={startEditing}>
                {layer.name}
              </Text>
            )}
            <Badge alignSelf="flex-start" colorPalette="gray" size="xs" variant="subtle">
              {t(layerBadgeKey(layer))}
            </Badge>
          </Stack>
          <Box flexShrink="0" onClick={stopPropagation} onPointerDown={stopPropagation}>
            <ToggleDot
              checked={layer.isEnabled}
              label={t('widgets.layers.actions.toggleVisibility')}
              onCheckedChange={handleToggleVisible}
            />
          </Box>
          <IconButton
            aria-label={t('widgets.layers.actions.toggleLock')}
            color={layer.isLocked ? 'fg' : 'fg.subtle'}
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

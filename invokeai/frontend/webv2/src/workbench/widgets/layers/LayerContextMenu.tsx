import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { CanvasLayerContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { LucideIcon } from 'lucide-react';
import type { ComponentProps, Dispatch } from 'react';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { deleteLayerActions, duplicateLayerActions } from '@workbench/canvasLayerOps';
import { IconButton, MenuContent, RenameDialog } from '@workbench/components/ui';
import {
  ArrowDownIcon,
  ArrowDownToLineIcon,
  ArrowUpIcon,
  ArrowUpToLineIcon,
  CopyIcon,
  EyeIcon,
  EyeOffIcon,
  ImageIcon,
  LockIcon,
  LockOpenIcon,
  MergeIcon,
  MoreVerticalIcon,
  PencilIcon,
  SlidersHorizontalIcon,
  Trash2Icon,
} from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerMoveKind } from './layerGroups';

import { getGroupPosition, reorderWithinGroupByKind } from './layerGroups';
import { resolveMenuTargetForRender } from './layerMenuState';
import {
  applyStructural,
  canConvertRasterControl,
  canMergeLayerDown,
  convertRasterControlLayer,
  createLayerId,
} from './layerOps';

type MenuPositioning = ComponentProps<typeof Menu.Root>['positioning'];
type MenuOpenChange = ComponentProps<typeof Menu.Root>['onOpenChange'];

const PANEL_POSITIONING: MenuPositioning = { placement: 'bottom-end' };

interface LayerMenuProps {
  dispatch: Dispatch<WorkbenchAction>;
  engine: CanvasEngine | null;
  index: number;
  layer: CanvasLayerContract;
  layers: readonly CanvasLayerContract[];
  /** Where the menu opens: the panel anchors to its trigger; the canvas uses a
   * virtual rect at the cursor. */
  positioning: MenuPositioning;
  /** Render the panel's ⋯ trigger button. Off in the canvas' controlled, anchored
   * mode, where there is no trigger DOM. */
  withTrigger?: boolean;
  /** Controlled open state (canvas right-click). Undefined ⇒ uncontrolled (panel). */
  open?: boolean;
  onOpenChange?: MenuOpenChange;
  lazyMount?: boolean;
  unmountOnExit?: boolean;
  /**
   * Controlled rename-in-flight state. When provided (canvas right-click), the
   * parent owns it so the rename dialog survives the menu closing (F1). Undefined
   * ⇒ the menu keeps this state internally (panel).
   */
  isRenaming?: boolean;
  onRenamingChange?: (isRenaming: boolean) => void;
}

/**
 * The shared layer context menu — ONE source of truth for the per-layer items and
 * their handlers, used by both the layers panel (a ⋯ trigger button) and the
 * canvas surface (right-click, anchored at the cursor). All actions operate on
 * `layer.id`, so they behave identically from either surface.
 *
 * Legacy-parity items (scoped to what webv2 supports today): rename, duplicate,
 * rasterize, raster↔control convert, group-aware z-arrange (move to front /
 * forward / backward / to back — within the layer's type group), merge-down,
 * visibility + lock toggles, and delete. Merge-down uses global z-adjacency
 * (compositing order); the arrange actions map to a splice inside the global array.
 *
 * The rename dialog is a SIBLING of `Menu.Root` (not inside its portal) so it
 * survives the menu closing when "Rename" is chosen.
 */
const LayerMenu = ({
  dispatch,
  engine,
  index,
  layer,
  layers,
  positioning,
  withTrigger,
  open,
  onOpenChange,
  lazyMount,
  unmountOnExit,
  isRenaming: controlledRenaming,
  onRenamingChange,
}: LayerMenuProps) => {
  const { t } = useTranslation();
  const [internalRenaming, setInternalRenaming] = useState(false);
  // Controlled (canvas) vs. uncontrolled (panel): the canvas parent owns the flag
  // so the rename dialog outlives the menu closing. The panel keeps it internally.
  const isRenaming = controlledRenaming ?? internalRenaming;
  const setRenaming = useCallback(
    (next: boolean) => {
      setInternalRenaming(next);
      onRenamingChange?.(next);
    },
    [onRenamingChange]
  );

  const groupPosition = getGroupPosition(layers, layer.id);
  const canMoveForward = !!groupPosition && groupPosition.index > 0;
  const canMoveBackward = !!groupPosition && groupPosition.index < groupPosition.count - 1;
  const canMerge = canMergeLayerDown(layers, index, !!engine);
  // "Rasterize layer" is offered only for parametric sources (shape/gradient/
  // text), where the engine can bake the params to pixels. A polygon shape has
  // no rasterizer yet, so it stays disabled.
  const canRasterize =
    !!engine &&
    layer.type === 'raster' &&
    (layer.source.type === 'gradient' ||
      layer.source.type === 'text' ||
      (layer.source.type === 'shape' && layer.source.kind !== 'polygon'));

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

  const reorder = useCallback(
    (kind: LayerMoveKind, label: string) => {
      const next = reorderWithinGroupByKind(layers, layer.id, kind);
      if (!next) {
        return;
      }
      applyStructural(
        engine,
        dispatch,
        label,
        { orderedIds: next, type: 'reorderCanvasLayers' },
        { orderedIds: layers.map((entry) => entry.id), type: 'reorderCanvasLayers' }
      );
    },
    [dispatch, engine, layer.id, layers]
  );

  const handleMoveToFront = useCallback(() => reorder('front', t('widgets.layers.actions.moveToFront')), [reorder, t]);
  const handleMoveForward = useCallback(
    () => reorder('forward', t('widgets.layers.actions.moveForward')),
    [reorder, t]
  );
  const handleMoveBackward = useCallback(
    () => reorder('backward', t('widgets.layers.actions.moveBackward')),
    [reorder, t]
  );
  const handleMoveToBack = useCallback(() => reorder('back', t('widgets.layers.actions.moveToBack')), [reorder, t]);

  const handleDuplicate = useCallback(() => {
    const { forward, inverse } = duplicateLayerActions(layer.id, createLayerId());
    applyStructural(engine, dispatch, t('widgets.layers.actions.duplicate'), forward, inverse);
  }, [dispatch, engine, layer.id, t]);

  const handleDelete = useCallback(() => {
    const { forward, inverse } = deleteLayerActions(layer, index);
    applyStructural(engine, dispatch, t('widgets.layers.actions.delete'), forward, inverse);
  }, [dispatch, engine, index, layer, t]);

  const handleMerge = useCallback(() => {
    // Pixel work: engine-only, and not recorded on the undo history.
    engine?.mergeLayerDown(layer.id);
  }, [engine, layer.id]);

  const handleRasterize = useCallback(() => {
    // Bakes the parametric source to pixels; the engine records ONE undoable
    // entry (inverse re-converts to the parametric source).
    engine?.rasterizeLayer(layer.id);
  }, [engine, layer.id]);

  const convert = useCallback(
    (targetType: 'raster' | 'control', label: string) => {
      const converted = convertRasterControlLayer(layer, targetType);
      if (!converted) {
        return;
      }
      // Convert in place, preserving the pixel source + id. The inverse restores
      // the layer verbatim (adapter/filter config and all).
      applyStructural(
        engine,
        dispatch,
        label,
        { id: layer.id, layer: converted, targetType, type: 'convertCanvasLayer' },
        { id: layer.id, layer: structuredClone(layer), targetType: layer.type, type: 'convertCanvasLayer' }
      );
    },
    [dispatch, engine, layer]
  );

  const handleConvertToControl = useCallback(
    () => convert('control', t('widgets.layers.actions.convertToControl')),
    [convert, t]
  );
  const handleConvertToRaster = useCallback(
    () => convert('raster', t('widgets.layers.actions.convertToRaster')),
    [convert, t]
  );
  // raster↔control conversion is offered only for pixel-backed layers (image/paint
  // raster sources, or a control layer). Parametric raster + mask layers cannot.
  const canConvertToControl = layer.type === 'raster' && canConvertRasterControl(layer);
  const canConvertToRaster = layer.type === 'control';

  const handleToggleVisibility = useCallback(() => {
    patchBase(
      t('widgets.layers.actions.toggleVisibility'),
      { isEnabled: !layer.isEnabled },
      { isEnabled: layer.isEnabled }
    );
  }, [layer.isEnabled, patchBase, t]);

  const handleToggleLock = useCallback(() => {
    patchBase(t('widgets.layers.actions.toggleLock'), { isLocked: !layer.isLocked }, { isLocked: layer.isLocked });
  }, [layer.isLocked, patchBase, t]);

  const openRename = useCallback(() => setRenaming(true), [setRenaming]);
  const closeRename = useCallback(() => setRenaming(false), [setRenaming]);
  const submitRename = useCallback(
    (name: string) => {
      patchBase(t('widgets.layers.actions.rename'), { name }, { name: layer.name });
    },
    [layer.name, patchBase, t]
  );

  return (
    <>
      <Menu.Root
        lazyMount={lazyMount}
        open={open}
        positioning={positioning}
        unmountOnExit={unmountOnExit}
        onOpenChange={onOpenChange}
      >
        {withTrigger ? (
          <Menu.Trigger asChild>
            <IconButton
              aria-label={t('widgets.layers.options')}
              color="fg.muted"
              size="2xs"
              variant="ghost"
              onClick={stopPropagation}
            >
              <MoreVerticalIcon />
            </IconButton>
          </Menu.Trigger>
        ) : null}
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="12rem">
              <LayerMenuItem
                icon={PencilIcon}
                label={t('widgets.layers.actions.rename')}
                value="rename"
                onSelect={openRename}
              />
              <LayerMenuItem
                icon={CopyIcon}
                label={t('widgets.layers.actions.duplicate')}
                value="duplicate"
                onSelect={handleDuplicate}
              />
              {canRasterize ? (
                <LayerMenuItem
                  icon={ImageIcon}
                  label={t('widgets.layers.actions.rasterize')}
                  value="rasterize"
                  onSelect={handleRasterize}
                />
              ) : null}
              {canConvertToControl ? (
                <LayerMenuItem
                  icon={SlidersHorizontalIcon}
                  label={t('widgets.layers.actions.convertToControl')}
                  value="convert-to-control"
                  onSelect={handleConvertToControl}
                />
              ) : null}
              {canConvertToRaster ? (
                <LayerMenuItem
                  icon={ImageIcon}
                  label={t('widgets.layers.actions.convertToRaster')}
                  value="convert-to-raster"
                  onSelect={handleConvertToRaster}
                />
              ) : null}
              <Menu.Separator borderColor="border.subtle" />
              <LayerMenuItem
                disabled={!canMoveForward}
                icon={ArrowUpToLineIcon}
                label={t('widgets.layers.actions.moveToFront')}
                value="move-to-front"
                onSelect={handleMoveToFront}
              />
              <LayerMenuItem
                disabled={!canMoveForward}
                icon={ArrowUpIcon}
                label={t('widgets.layers.actions.moveForward')}
                value="move-forward"
                onSelect={handleMoveForward}
              />
              <LayerMenuItem
                disabled={!canMoveBackward}
                icon={ArrowDownIcon}
                label={t('widgets.layers.actions.moveBackward')}
                value="move-backward"
                onSelect={handleMoveBackward}
              />
              <LayerMenuItem
                disabled={!canMoveBackward}
                icon={ArrowDownToLineIcon}
                label={t('widgets.layers.actions.moveToBack')}
                value="move-to-back"
                onSelect={handleMoveToBack}
              />
              <Menu.Separator borderColor="border.subtle" />
              <LayerMenuItem
                disabled={!canMerge}
                icon={MergeIcon}
                label={t('widgets.layers.actions.mergeDown')}
                value="merge-down"
                onSelect={handleMerge}
              />
              <LayerMenuItem
                icon={layer.isEnabled ? EyeOffIcon : EyeIcon}
                label={layer.isEnabled ? t('widgets.layers.actions.hide') : t('widgets.layers.actions.show')}
                value="toggle-visibility"
                onSelect={handleToggleVisibility}
              />
              <LayerMenuItem
                icon={layer.isLocked ? LockOpenIcon : LockIcon}
                label={layer.isLocked ? t('widgets.layers.actions.unlock') : t('widgets.layers.actions.lock')}
                value="toggle-lock"
                onSelect={handleToggleLock}
              />
              <Menu.Separator borderColor="border.subtle" />
              <LayerMenuItem
                color="fg.error"
                icon={Trash2Icon}
                label={t('widgets.layers.actions.delete')}
                value="delete"
                onSelect={handleDelete}
              />
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <RenameDialog
        initialName={layer.name}
        isOpen={isRenaming}
        label={t('widgets.layers.actions.rename')}
        submitLabel={t('widgets.layers.actions.rename')}
        title={t('widgets.layers.actions.rename')}
        onClose={closeRename}
        onSubmit={submitRename}
      />
    </>
  );
};

interface LayerRowMenuProps {
  dispatch: Dispatch<WorkbenchAction>;
  engine: CanvasEngine | null;
  index: number;
  layer: CanvasLayerContract;
  layers: readonly CanvasLayerContract[];
}

/** The layers-panel per-row context menu: a ⋯ trigger button, opened below it. */
export const LayerContextMenu = (props: LayerRowMenuProps) => (
  <LayerMenu {...props} positioning={PANEL_POSITIONING} withTrigger />
);

/** The layer + pointer position a canvas right-click resolved to. */
export interface CanvasLayerContextMenuTarget {
  layerId: string;
  x: number;
  y: number;
}

/**
 * The canvas-surface right-click menu: the SAME {@link LayerMenu}, anchored at the
 * cursor via a 1×1 virtual rect (no trigger DOM), controlled by `target`. The
 * canvas widget sets `target` to the hit layer + pointer position after selecting
 * it; `null` closes the menu. The layer and its global index are resolved from
 * `target.layerId` against the live layer list, so the shared items get the exact
 * same inputs the panel passes. Keyed by layer id so switching target resets the
 * menu's rename state.
 *
 * Choosing "Rename" closes the menu, which nulls `target`. The rename dialog lives
 * inside {@link LayerMenu} (a sibling of the menu), so it must survive that: the
 * wrapper owns the rename-in-flight flag and keeps rendering against the last-known
 * (sticky) target until the dialog closes (F1).
 */
export const CanvasLayerContextMenu = ({
  dispatch,
  engine,
  layers,
  target,
  onClose,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  engine: CanvasEngine | null;
  layers: readonly CanvasLayerContract[];
  target: CanvasLayerContextMenuTarget | null;
  onClose: () => void;
}) => {
  // The layer a pending rename is anchored to. Captured when the dialog opens (the
  // live `target` is still set then) and cleared when it closes — set only inside
  // the rename callback, never during render, so the rename dialog survives the menu
  // closing (which nulls `target`).
  const [renameTarget, setRenameTarget] = useState<CanvasLayerContextMenuTarget | null>(null);
  const renderTarget = resolveMenuTargetForRender(target, renameTarget);

  const index = renderTarget ? layers.findIndex((entry) => entry.id === renderTarget.layerId) : -1;
  const layer = index >= 0 ? layers[index] : undefined;

  const anchorX = renderTarget?.x ?? 0;
  const anchorY = renderTarget?.y ?? 0;
  const positioning = useMemo<MenuPositioning>(
    () => ({
      getAnchorRect: () => ({ height: 1, width: 1, x: anchorX, y: anchorY }),
      placement: 'bottom-start',
    }),
    [anchorX, anchorY]
  );
  const handleOpenChange = useCallback(
    (details: { open: boolean }) => {
      if (!details.open) {
        onClose();
      }
    },
    [onClose]
  );
  const handleRenamingChange = useCallback(
    (renaming: boolean) => {
      setRenameTarget(renaming ? target : null);
    },
    [target]
  );

  if (!renderTarget || !layer) {
    return null;
  }

  return (
    <LayerMenu
      key={renderTarget.layerId}
      dispatch={dispatch}
      engine={engine}
      index={index}
      isRenaming={renameTarget !== null}
      layer={layer}
      layers={layers}
      lazyMount
      // The menu itself is visible only while the live target is set; once "Rename"
      // closes it (target → null), the subtree stays mounted for the dialog.
      open={!!target}
      positioning={positioning}
      unmountOnExit
      onOpenChange={handleOpenChange}
      onRenamingChange={handleRenamingChange}
    />
  );
};

const stopPropagation = (event: { stopPropagation: () => void }): void => event.stopPropagation();

const LayerMenuItem = ({
  color,
  disabled,
  icon,
  label,
  onSelect,
  value,
}: {
  color?: string;
  disabled?: boolean;
  icon: LucideIcon;
  label: string;
  onSelect: () => void;
  value: string;
}) => (
  <Menu.Item color={color} disabled={disabled} value={value} onSelect={onSelect}>
    <HStack gap="2" minW="0" w="full">
      <Icon as={icon} boxSize="3.5" color={color ?? 'fg.subtle'} flexShrink={0} />
      <Text flex="1" fontSize="xs">
        {label}
      </Text>
    </HStack>
  </Menu.Item>
);

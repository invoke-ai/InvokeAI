import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { GenerateReferenceImage } from '@workbench/generation/types';
import type { CanvasLayerContract, CanvasMaskContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { LucideIcon } from 'lucide-react';
import type { ComponentProps, Dispatch } from 'react';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { deleteLayerActions, duplicateLayerActions } from '@workbench/canvasLayerOps';
import { IconButton, MenuContent, RenameDialog } from '@workbench/components/ui';
import { uploadGalleryImage } from '@workbench/gallery/api';
import { useModelsSelector } from '@workbench/models/modelsStore';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import {
  ArrowDownIcon,
  ArrowDownToLineIcon,
  ArrowUpIcon,
  ArrowUpToLineIcon,
  CopyIcon,
  CropIcon,
  EyeIcon,
  EyeOffIcon,
  ImageIcon,
  LockIcon,
  LockOpenIcon,
  MergeIcon,
  MoreVerticalIcon,
  PencilIcon,
  SaveIcon,
  SlidersHorizontalIcon,
  Trash2Icon,
} from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerMoveKind } from './layerGroups';

import { getLayerContextActions, type LayerContextAction, type LayerContextActionId } from './layerContextActions';
import { copyBlobToClipboard } from './layerExportActions';
import { reorderWithinGroupByKind } from './layerGroups';
import { resolveMenuTargetForRender } from './layerMenuState';
import {
  applyStructural,
  convertRasterControlLayer,
  createLayerId,
  fitLayerTransformToBbox,
  getControlTransparencyEffectPatch,
  getInpaintDenoiseLimitPatch,
  getInpaintNoisePatch,
  getRegionalGuidanceAutoNegativePatch,
  getRegionalGuidanceNegativePromptPatch,
  getRegionalGuidancePositivePromptPatch,
  getRegionalGuidanceReferenceImagePatch,
} from './layerOps';

type MenuPositioning = ComponentProps<typeof Menu.Root>['positioning'];
type MenuOpenChange = ComponentProps<typeof Menu.Root>['onOpenChange'];

type LayerConfigPatch =
  | { layerType: 'control'; withTransparencyEffect?: boolean }
  | {
      layerType: 'regional_guidance';
      mask?: Partial<CanvasMaskContract>;
      positivePrompt?: string | null;
      negativePrompt?: string | null;
      autoNegative?: boolean;
      referenceImages?: GenerateReferenceImage[];
    }
  | { layerType: 'inpaint_mask'; noiseLevel?: number; denoiseLimit?: number };

const PANEL_POSITIONING: MenuPositioning = { placement: 'bottom-end' };

/** The main model's base, read from the generate widget values (drives regional ref-image support). */
const useSelectedModelBase = (): string | null => {
  const modelKey = useActiveProjectSelector((project) => {
    const values = getProjectWidgetValues(project, 'generate');
    const model = values?.model;
    return model && typeof model === 'object' && 'key' in model ? String((model as { key: unknown }).key) : null;
  });
  const models = useModelsSelector((snapshot) => snapshot.models);
  return useMemo(() => models.find((model) => model.key === modelKey)?.base ?? null, [models, modelKey]);
};

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
  const base = useSelectedModelBase();
  const { bbox, documentRect } = useActiveProjectSelector((project) => ({
    bbox: project.canvas.document.bbox,
    documentRect: { height: project.canvas.document.height, width: project.canvas.document.width, x: 0, y: 0 },
  }));
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

  const patchConfig = useCallback(
    (label: string, forward: LayerConfigPatch, inverse: LayerConfigPatch) => {
      applyStructural(
        engine,
        dispatch,
        label,
        { config: forward, id: layer.id, type: 'updateCanvasLayerConfig' },
        { config: inverse, id: layer.id, type: 'updateCanvasLayerConfig' }
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

  const actions = useMemo(
    () => getLayerContextActions({ hasEngine: !!engine, index, layer, layers }),
    [engine, index, layer, layers]
  );
  const iconActions = useMemo(() => actions.filter((action) => action.group === 'icons'), [actions]);
  const actionGroups = useMemo(
    () =>
      LAYER_ACTION_GROUPS.map((group) => ({
        actions: actions.filter((action) => action.group === group),
        group,
      })).filter((entry) => entry.actions.length > 0),
    [actions]
  );

  const getActionLabel = useCallback(
    (id: LayerContextActionId) => {
      const action = actions.find((entry) => entry.id === id);
      return action ? t(action.labelKey, { defaultValue: action.defaultLabel }) : id;
    },
    [actions, t]
  );

  const handleTransform = useCallback(() => {
    dispatch({ id: layer.id, type: 'setCanvasSelectedLayer' });
    engine?.setTool('transform');
  }, [dispatch, engine, layer.id]);

  const handleFitToBbox = useCallback(() => {
    const transform = fitLayerTransformToBbox(layer, bbox, documentRect);
    if (!transform) {
      return;
    }
    patchBase(getActionLabel('fit-to-bbox'), { transform }, { transform: layer.transform });
  }, [bbox, documentRect, getActionLabel, layer, patchBase]);

  const handleSaveToAssets = useCallback(async () => {
    const result = await engine?.exportBakedLayerBlob(layer.id, { includeDisabled: true });
    if (!result || result.status !== 'ok') {
      return;
    }
    const file = new File([result.blob], `layer-${layer.id}.png`, { type: result.blob.type || 'image/png' });
    await uploadGalleryImage(file, 'none');
  }, [engine, layer.id]);

  const handleCopyToClipboard = useCallback(async () => {
    const result = await engine?.exportBakedLayerBlob(layer.id, { includeDisabled: true });
    if (!result || result.status !== 'ok') {
      return;
    }
    await copyBlobToClipboard(result.blob);
  }, [engine, layer.id]);

  const handleCropToBbox = useCallback(() => {
    void engine?.cropLayerToBbox(layer.id);
  }, [engine, layer.id]);

  const handleCopyToRaster = useCallback(() => {
    void engine?.copyLayerToRaster(layer.id);
  }, [engine, layer.id]);

  const handleLayerConfigAction = useCallback(
    (id: LayerContextActionId) => {
      if (id === 'control-transparency-effect' && layer.type === 'control') {
        const { forward, inverse } = getControlTransparencyEffectPatch(layer);
        patchConfig(getActionLabel(id), forward, inverse);
      } else if (id === 'regional-positive-prompt' && layer.type === 'regional_guidance') {
        const { forward, inverse } = getRegionalGuidancePositivePromptPatch(layer);
        patchConfig(getActionLabel(id), forward, inverse);
      } else if (id === 'regional-negative-prompt' && layer.type === 'regional_guidance') {
        const { forward, inverse } = getRegionalGuidanceNegativePromptPatch(layer);
        patchConfig(getActionLabel(id), forward, inverse);
      } else if (id === 'regional-reference-image' && layer.type === 'regional_guidance') {
        const { forward, inverse } = getRegionalGuidanceReferenceImagePatch(layer, base);
        patchConfig(getActionLabel(id), forward, inverse);
      } else if (id === 'regional-auto-negative' && layer.type === 'regional_guidance') {
        const { forward, inverse } = getRegionalGuidanceAutoNegativePatch(layer);
        patchConfig(getActionLabel(id), forward, inverse);
      } else if (id === 'inpaint-noise' && layer.type === 'inpaint_mask') {
        const { forward, inverse } = getInpaintNoisePatch(layer);
        patchConfig(getActionLabel(id), forward, inverse);
      } else if (id === 'inpaint-denoise-limit' && layer.type === 'inpaint_mask') {
        const { forward, inverse } = getInpaintDenoiseLimitPatch(layer);
        patchConfig(getActionLabel(id), forward, inverse);
      }
    },
    [base, getActionLabel, layer, patchConfig]
  );

  const handleAction = useCallback(
    (id: LayerContextActionId) => {
      switch (id) {
        case 'move-to-front':
          handleMoveToFront();
          break;
        case 'move-forward':
          handleMoveForward();
          break;
        case 'move-backward':
          handleMoveBackward();
          break;
        case 'move-to-back':
          handleMoveToBack();
          break;
        case 'duplicate':
          handleDuplicate();
          break;
        case 'rename':
          openRename();
          break;
        case 'transform':
          handleTransform();
          break;
        case 'fit-to-bbox':
          handleFitToBbox();
          break;
        case 'save-to-assets':
          void handleSaveToAssets();
          break;
        case 'copy-to-clipboard':
          void handleCopyToClipboard();
          break;
        case 'crop-to-bbox':
          handleCropToBbox();
          break;
        case 'copy-to-raster':
          handleCopyToRaster();
          break;
        case 'rasterize':
          handleRasterize();
          break;
        case 'convert-to-control':
          handleConvertToControl();
          break;
        case 'convert-to-raster':
          handleConvertToRaster();
          break;
        case 'merge-down':
          handleMerge();
          break;
        case 'toggle-visibility':
          handleToggleVisibility();
          break;
        case 'toggle-lock':
          handleToggleLock();
          break;
        case 'delete':
          handleDelete();
          break;
        default:
          handleLayerConfigAction(id);
          break;
      }
    },
    [
      handleConvertToControl,
      handleConvertToRaster,
      handleCopyToClipboard,
      handleCopyToRaster,
      handleCropToBbox,
      handleDelete,
      handleDuplicate,
      handleFitToBbox,
      handleLayerConfigAction,
      handleMerge,
      handleMoveBackward,
      handleMoveForward,
      handleMoveToBack,
      handleMoveToFront,
      handleRasterize,
      handleToggleLock,
      handleToggleVisibility,
      handleTransform,
      handleSaveToAssets,
      openRename,
    ]
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
            <MenuContent minW="14rem" py="1">
              <HStack gap="1">
                {iconActions.map((action) => (
                  <LayerMenuIconActionItem key={action.id} action={action} handleAction={handleAction} t={t} />
                ))}
              </HStack>
              {actionGroups.map(({ actions, group }) => (
                <LayerMenuGroup key={group} actions={actions} handleAction={handleAction} t={t} />
              ))}
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

const LAYER_ACTION_GROUPS: readonly LayerContextAction['group'][] = [
  'edit',
  'convert',
  'layerConfig',
  'state',
  'danger',
];

const LAYER_ACTION_ICONS: Record<LayerContextActionId, LucideIcon> = {
  'control-transparency-effect': SlidersHorizontalIcon,
  'copy-to-clipboard': CopyIcon,
  'copy-to-raster': CopyIcon,
  'crop-to-bbox': CropIcon,
  'convert-to-control': SlidersHorizontalIcon,
  'convert-to-raster': ImageIcon,
  delete: Trash2Icon,
  duplicate: CopyIcon,
  'fit-to-bbox': ImageIcon,
  'inpaint-denoise-limit': SlidersHorizontalIcon,
  'inpaint-noise': SlidersHorizontalIcon,
  'merge-down': MergeIcon,
  'move-backward': ArrowDownIcon,
  'move-forward': ArrowUpIcon,
  'move-to-back': ArrowDownToLineIcon,
  'move-to-front': ArrowUpToLineIcon,
  rasterize: ImageIcon,
  'regional-auto-negative': SlidersHorizontalIcon,
  'regional-negative-prompt': PencilIcon,
  'regional-positive-prompt': PencilIcon,
  'regional-reference-image': ImageIcon,
  rename: PencilIcon,
  'save-to-assets': SaveIcon,
  'toggle-lock': LockIcon,
  'toggle-visibility': EyeIcon,
  transform: SlidersHorizontalIcon,
};

const stopPropagation = (event: { stopPropagation: () => void }): void => event.stopPropagation();

const LayerMenuGroup = ({
  actions,
  handleAction,
  t,
}: {
  actions: readonly LayerContextAction[];
  handleAction: (id: LayerContextActionId) => void;
  t: (key: string, options: { defaultValue: string }) => string;
}) => (
  <>
    <Menu.Separator borderColor="border.subtle" />
    {actions.map((action) => (
      <LayerMenuActionItem key={action.id} action={action} handleAction={handleAction} t={t} />
    ))}
  </>
);

const LayerMenuIconActionItem = ({
  action,
  handleAction,
  t,
}: {
  action: LayerContextAction;
  handleAction: (id: LayerContextActionId) => void;
  t: (key: string, options: { defaultValue: string }) => string;
}) => {
  const onSelect = useCallback(() => handleAction(action.id), [action.id, handleAction]);

  return (
    <LayerMenuIconItem
      disabled={action.isDisabled}
      icon={LAYER_ACTION_ICONS[action.id]}
      label={t(action.labelKey, { defaultValue: action.defaultLabel })}
      value={action.id}
      onSelect={onSelect}
    />
  );
};

const LayerMenuActionItem = ({
  action,
  handleAction,
  t,
}: {
  action: LayerContextAction;
  handleAction: (id: LayerContextActionId) => void;
  t: (key: string, options: { defaultValue: string }) => string;
}) => {
  const onSelect = useCallback(() => handleAction(action.id), [action.id, handleAction]);

  return (
    <LayerMenuItem
      color={action.tone === 'danger' ? 'fg.error' : undefined}
      disabled={action.isDisabled}
      icon={getLayerActionIcon(action)}
      label={t(action.labelKey, { defaultValue: action.defaultLabel })}
      value={action.id}
      onSelect={onSelect}
    />
  );
};

const getLayerActionIcon = (action: LayerContextAction): LucideIcon => {
  if (action.id === 'toggle-visibility') {
    return action.defaultLabel === 'Hide' ? EyeOffIcon : EyeIcon;
  }
  if (action.id === 'toggle-lock') {
    return action.defaultLabel === 'Unlock' ? LockOpenIcon : LockIcon;
  }
  return LAYER_ACTION_ICONS[action.id];
};

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

const LayerMenuIconItem = ({
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
  <Menu.Item
    aria-label={label}
    color={color}
    disabled={disabled}
    flex="1"
    justifyContent="center"
    value={value}
    onSelect={onSelect}
  >
    <Icon as={icon} boxSize="4" color={color ?? 'fg.subtle'} />
  </Menu.Item>
);

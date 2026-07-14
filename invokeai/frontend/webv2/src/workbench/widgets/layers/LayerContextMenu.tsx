import type { BooleanRasterOperation } from '@workbench/canvas-engine/engine';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { GenerateReferenceImage } from '@workbench/generation/types';
import type { CanvasDocumentContractV2, CanvasLayerContract, CanvasMaskContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { LucideIcon } from 'lucide-react';
import type { ComponentProps, Dispatch } from 'react';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { getCanvasOperations } from '@workbench/canvas-operations/createCanvasEngine';
import { deleteLayerActions, duplicateLayerActions } from '@workbench/canvasLayerOps';
import { IconButton, MenuContent, RenameDialog } from '@workbench/components/ui';
import { uploadGalleryImage } from '@workbench/gallery/api';
import { useNotify } from '@workbench/useNotify';
import { isCanvasInteractionLocked } from '@workbench/widgets/canvas/canvasInteractionLock';
import { useCanvasDocumentEditingLocked, useLayerThumbnailVersion } from '@workbench/widgets/canvas/engineStoreHooks';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import {
  ArrowRightLeftIcon,
  ArrowUpDownIcon,
  ChevronRightIcon,
  CopyIcon,
  MergeIcon,
  MoreVerticalIcon,
  PlusIcon,
} from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerContextMenuItem, LayerContextMenuSection, LayerContextSubmenuId } from './layerContextMenuLayout';
import type { LayerMoveKind } from './layerGroups';
import type { LayerMenuDialogKind, LayerMenuDialogState } from './layerMenuState';
import type { LayerPropertiesSection } from './layerPropertiesRequestStore';

import {
  getLayerContextActions,
  type LayerConfigPatchKind,
  type LayerContextAction,
  type LayerContextActionEffects,
  type LayerContextActionId,
  type LayerContextActionState,
  type LayerType,
} from './layerContextActions';
import { getLayerContextMenuLayout } from './layerContextMenuLayout';
import { copyBlobToClipboard } from './layerExportActions';
import { reorderWithinGroupByKind } from './layerGroups';
import { resolveMenuTargetForRender } from './layerMenuState';
import {
  applyStructural,
  convertRasterToControl,
  convertRasterToInpaintMask,
  convertRasterToRegionalGuidance,
  convertRasterControlLayer,
  copyControlToInpaintMask,
  copyControlToRaster,
  copyControlToRegionalGuidance,
  copyMaskToRegionalGuidance,
  copyRasterToControl,
  copyRasterToInpaintMask,
  copyRasterToRegionalGuidance,
  copyRegionalGuidanceToInpaintMask,
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
import { requestLayerProperties } from './layerPropertiesRequestStore';
import { RunLayerWorkflowDialog, useLayerWorkflowAvailability } from './RunLayerWorkflowDialog';
import { useSelectedModelBase } from './useSelectedModelBase';

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

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

const assertNever = (value: never): never => {
  throw new Error(`Unhandled layer action result: ${String(value)}`);
};

type LayerActionErrorStatus = 'busy' | 'disabled' | 'empty' | 'locked' | 'missing' | 'not-ready' | 'unsupported';

const LAYER_ACTION_ERROR_KEYS: Record<LayerActionErrorStatus, string> = {
  busy: 'widgets.layers.actions.busy',
  disabled: 'widgets.layers.actions.disabled',
  empty: 'widgets.layers.actions.empty',
  locked: 'widgets.layers.actions.locked',
  missing: 'widgets.layers.actions.missing',
  'not-ready': 'widgets.layers.actions.notReady',
  unsupported: 'widgets.layers.actions.unsupported',
};

const hasPureExportableLayerContent = (layer: CanvasLayerContract, document: CanvasDocumentContractV2): boolean => {
  if (!renderableSourceOf(layer)) {
    return false;
  }
  const contentRect = getSourceContentRect(layer, document);
  return contentRect.width > 0 && contentRect.height > 0;
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
   * Controlled sibling-dialog state. When provided (canvas right-click), the
   * parent owns it so dialogs survive the menu closing. Undefined means the menu
   * keeps this state internally (panel).
   */
  dialogKind?: LayerMenuDialogKind | null;
  onDialogKindChange?: (kind: LayerMenuDialogKind | null) => void;
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
 * Sibling dialogs live beside `Menu.Root` (not inside its portal) so they
 * survive the menu closing after their action is chosen.
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
  dialogKind: controlledDialogKind,
  onDialogKindChange,
}: LayerMenuProps) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const base = useSelectedModelBase();
  const workflowAvailability = useLayerWorkflowAvailability();
  const canvas = useActiveProjectSelector((project) => project.canvas);
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const { document } = canvas;
  const { bbox } = document;
  const documentRect = useMemo(
    () => ({ height: document.height, width: document.width, x: 0, y: 0 }),
    [document.height, document.width]
  );
  const documentEditingLocked = useCanvasDocumentEditingLocked(engine);
  const interactionLocked = isCanvasInteractionLocked(canvas, queueItems) || documentEditingLocked;
  // Re-render when live, not-yet-persisted paint/mask pixels change.
  useLayerThumbnailVersion(engine, layer.id);
  const hasSupportedContent = engine
    ? engine.exports.hasExportableLayerContent(layer.id)
    : hasPureExportableLayerContent(layer, document);
  const [internalDialogKind, setInternalDialogKind] = useState<LayerMenuDialogKind | null>(null);
  // Controlled (canvas) vs. uncontrolled (panel): the canvas parent owns the
  // dialog kind so its sibling survives menu close; panel rows keep it locally.
  const dialogKind = controlledDialogKind !== undefined ? controlledDialogKind : internalDialogKind;
  const setDialogKind = useCallback(
    (next: LayerMenuDialogKind | null) => {
      setInternalDialogKind(next);
      onDialogKindChange?.(next);
    },
    [onDialogKindChange]
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

  const actionState = useMemo<LayerContextActionState>(
    () => ({
      canRunWorkflow: workflowAvailability.canRunWorkflow,
      document,
      hasEngine: engine !== null,
      hasSupportedContent,
      hasWorkflowBindings: workflowAvailability.hasWorkflowBindings,
      index,
      interactionLocked,
      layer,
    }),
    [
      document,
      engine,
      hasSupportedContent,
      index,
      interactionLocked,
      layer,
      workflowAvailability.canRunWorkflow,
      workflowAvailability.hasWorkflowBindings,
    ]
  );
  const actions = useMemo(() => getLayerContextActions(actionState), [actionState]);
  const menuLayout = useMemo(() => getLayerContextMenuLayout(actions), [actions]);

  const getActionLabel = useCallback(
    (id: LayerContextActionId) => {
      const action = actions.find((entry) => entry.id === id);
      return action ? t(action.labelKey, { defaultValue: action.defaultLabel }) : id;
    },
    [actions, t]
  );

  const makeStatusError = useCallback(
    (status: LayerActionErrorStatus): Error => new Error(t(LAYER_ACTION_ERROR_KEYS[status])),
    [t]
  );

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
    engine?.layers.mergeLayerDown(layer.id);
  }, [engine, layer.id]);

  const handleRasterize = useCallback(() => {
    // Bakes the parametric source to pixels; the engine records ONE undoable
    // entry (inverse re-converts to the parametric source).
    engine?.layers.rasterizeLayer(layer.id);
  }, [engine, layer.id]);

  const addCopy = useCallback(
    (copied: CanvasLayerContract | null, label: string) => {
      if (!copied) {
        throw new Error(t('widgets.layers.actions.copyFailed'));
      }
      if (engine) {
        if (!engine.layers.commitLayerCopy(label, layer.id, copied, index)) {
          throw new Error(t('widgets.layers.actions.copyFailed'));
        }
        return;
      }
      applyStructural(
        engine,
        dispatch,
        label,
        { index, layer: copied, type: 'addCanvasLayer' },
        { ids: [copied.id], type: 'removeCanvasLayers' }
      );
    },
    [dispatch, engine, index, layer.id, t]
  );

  const convert = useCallback(
    (targetType: CanvasLayerContract['type'], label: string) => {
      const converted =
        layer.type === 'raster' && targetType === 'control'
          ? convertRasterToControl(layer, base)
          : layer.type === 'raster' && targetType === 'inpaint_mask'
            ? convertRasterToInpaintMask(layer)
            : layer.type === 'raster' && targetType === 'regional_guidance'
              ? convertRasterToRegionalGuidance(layer)
              : targetType === 'raster'
                ? convertRasterControlLayer(layer, 'raster')
                : null;
      if (!converted) {
        throw makeStatusError('unsupported');
      }
      if (engine) {
        // Pass the immutable live object: the engine rejects stale menu actions
        // by identity and clones the inverse contract internally.
        if (!engine.layers.commitLayerConversion(label, layer, converted)) {
          throw makeStatusError('not-ready');
        }
      } else {
        // Convert in place, preserving the pixel source + id. The inverse restores
        // the layer verbatim (adapter/filter config and all).
        const original = structuredClone(layer);
        applyStructural(
          engine,
          dispatch,
          label,
          { id: layer.id, layer: converted, targetType, type: 'convertCanvasLayer' },
          { id: layer.id, layer: original, targetType: layer.type, type: 'convertCanvasLayer' }
        );
      }
    },
    [base, dispatch, engine, layer, makeStatusError]
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

  const openRename = useCallback(() => setDialogKind('rename'), [setDialogKind]);
  const closeDialog = useCallback(() => setDialogKind(null), [setDialogKind]);
  const openRunWorkflow = useCallback(() => setDialogKind('run-workflow'), [setDialogKind]);
  const startSelectObject = useCallback(
    (layerId: string) => {
      if (!engine) {
        throw makeStatusError('not-ready');
      }
      const result = getCanvasOperations(engine).startSelectObject(layerId);
      if (result !== 'started') {
        throw makeStatusError(result);
      }
    },
    [engine, makeStatusError]
  );
  const startFilter = useCallback(
    (layerId: string) => {
      if (!engine) {
        throw makeStatusError('not-ready');
      }
      const result = getCanvasOperations(engine).startFilterOperation(layerId);
      if (result !== 'started') {
        throw makeStatusError(result);
      }
    },
    [engine, makeStatusError]
  );
  const submitRename = useCallback(
    (name: string) => {
      patchBase(t('widgets.layers.actions.rename'), { name }, { name: layer.name });
    },
    [layer.name, patchBase, t]
  );

  const handleTransform = useCallback(() => {
    dispatch({ id: layer.id, type: 'setCanvasSelectedLayer' });
    engine?.tools.setTool('transform');
  }, [dispatch, engine, layer.id]);

  const handleFitToBbox = useCallback(() => {
    const transform = fitLayerTransformToBbox(layer, bbox, documentRect);
    if (!transform) {
      throw makeStatusError('empty');
    }
    patchBase(getActionLabel('fit-to-bbox'), { transform }, { transform: layer.transform });
  }, [bbox, documentRect, getActionLabel, layer, makeStatusError, patchBase]);

  const handleSaveToAssets = useCallback(async () => {
    if (!engine) {
      throw makeStatusError('not-ready');
    }
    const result = await engine.exports.exportBakedLayerBlob(layer.id, { includeDisabled: true });
    if (result.status !== 'ok') {
      throw makeStatusError(result.status);
    }
    const file = new File([result.blob], `layer-${layer.id}.png`, { type: result.blob.type || 'image/png' });
    await uploadGalleryImage(file, 'none');
  }, [engine, layer.id, makeStatusError]);

  const handleCopyToClipboard = useCallback(async () => {
    if (!engine) {
      throw makeStatusError('not-ready');
    }
    const result = await engine.exports.exportBakedLayerBlob(layer.id, { includeDisabled: true });
    if (result.status !== 'ok') {
      throw makeStatusError(result.status);
    }
    await copyBlobToClipboard(result.blob);
  }, [engine, layer.id, makeStatusError]);

  const handleCropToBbox = useCallback(async () => {
    if (!engine) {
      throw makeStatusError('not-ready');
    }
    const result = await engine.layers.cropLayerToBbox(layer.id);
    switch (result.status) {
      case 'cropped':
        notify.success(t('widgets.layers.actions.cropped'));
        return;
      case 'missing':
      case 'locked':
      case 'empty':
      case 'not-ready':
        throw makeStatusError(result.status);
      case 'busy':
        throw new Error(t('widgets.layers.actions.cropBusy'));
      case 'unsupported':
        throw new Error(t('widgets.layers.actions.cropUnsupported'));
      case 'failed':
        throw new Error(`${t('widgets.layers.actions.cropFailed')} ${result.message}`);
      default:
        return assertNever(result);
    }
  }, [engine, layer.id, makeStatusError, notify, t]);

  const handleExtractMaskedArea = useCallback(async () => {
    if (!engine) {
      throw makeStatusError('not-ready');
    }
    const result = await engine.exports.extractMaskedArea(layer.id);
    if (result.status !== 'extracted') {
      throw makeStatusError(result.status);
    }
  }, [engine, layer.id, makeStatusError]);

  const handleOpenProperties = useCallback(
    (section: LayerPropertiesSection) => {
      dispatch({ region: 'right', type: 'openRegionWidget', widgetId: 'layers' });
      requestLayerProperties(layer.id, section);
    },
    [dispatch, layer.id]
  );

  const handleBooleanRaster = useCallback(
    async (operation: BooleanRasterOperation) => {
      if (!engine) {
        throw makeStatusError('not-ready');
      }
      const result = await engine.layers.booleanMergeRasterLayers(layer.id, operation);
      if (result !== 'merged') {
        throw makeStatusError(result);
      }
    },
    [engine, layer.id, makeStatusError]
  );

  const handleCopyToRaster = useCallback(async () => {
    if (layer.type === 'control') {
      addCopy(copyControlToRaster(layer, createLayerId()), getActionLabel('copy-to-raster'));
      return;
    }
    if (!engine) {
      throw makeStatusError('not-ready');
    }
    if ((await engine.layers.copyLayerToRaster(layer.id)) === null) {
      throw new Error(t('widgets.layers.actions.copyFailed'));
    }
  }, [addCopy, engine, getActionLabel, layer, makeStatusError, t]);

  const handleCopyToControl = useCallback(() => {
    if (layer.type === 'raster') {
      addCopy(copyRasterToControl(layer, createLayerId(), base), getActionLabel('copy-to-control'));
    }
  }, [addCopy, base, getActionLabel, layer]);

  const handleCopyToInpaintMask = useCallback(() => {
    const id = createLayerId();
    const copied =
      layer.type === 'raster'
        ? copyRasterToInpaintMask(layer, id)
        : layer.type === 'control'
          ? copyControlToInpaintMask(layer, id)
          : layer.type === 'regional_guidance'
            ? copyRegionalGuidanceToInpaintMask(layer, id)
            : null;
    addCopy(copied, getActionLabel('copy-to-inpaint-mask'));
  }, [addCopy, getActionLabel, layer]);

  const handleCopyToRegionalGuidance = useCallback(() => {
    const id = createLayerId();
    const copied =
      layer.type === 'raster'
        ? copyRasterToRegionalGuidance(layer, id)
        : layer.type === 'control'
          ? copyControlToRegionalGuidance(layer, id)
          : layer.type === 'inpaint_mask'
            ? copyMaskToRegionalGuidance(layer, id)
            : null;
    addCopy(copied, getActionLabel('copy-to-regional-guidance'));
  }, [addCopy, getActionLabel, layer]);

  const handleCopyTo = useCallback(
    (target: LayerType): void | Promise<void> => {
      switch (target) {
        case 'raster':
          return handleCopyToRaster();
        case 'control':
          return handleCopyToControl();
        case 'inpaint_mask':
          return handleCopyToInpaintMask();
        case 'regional_guidance':
          return handleCopyToRegionalGuidance();
      }
    },
    [handleCopyToControl, handleCopyToInpaintMask, handleCopyToRaster, handleCopyToRegionalGuidance]
  );

  const handleLayerConfigAction = useCallback(
    (id: LayerConfigPatchKind) => {
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

  const effects = useMemo<LayerContextActionEffects>(
    () => ({
      booleanMerge: handleBooleanRaster,
      convertTo: (target) => {
        const actionId: LayerContextActionId =
          target === 'control'
            ? 'convert-to-control'
            : target === 'raster'
              ? 'convert-to-raster'
              : target === 'inpaint_mask'
                ? 'convert-to-inpaint-mask'
                : 'convert-to-regional-guidance';
        convert(target, getActionLabel(actionId));
      },
      copyTo: handleCopyTo,
      copyToClipboard: handleCopyToClipboard,
      cropToBbox: handleCropToBbox,
      delete: handleDelete,
      duplicate: handleDuplicate,
      extractMaskedArea: handleExtractMaskedArea,
      fitToBbox: handleFitToBbox,
      mergeDown: handleMerge,
      openProperties: handleOpenProperties,
      openRename,
      openRunWorkflow,
      startSelectObject,
      startFilter,
      patchConfig: handleLayerConfigAction,
      rasterize: handleRasterize,
      reorder: (kind, actionId) => reorder(kind, getActionLabel(actionId)),
      saveToAssets: handleSaveToAssets,
      toggleLock: handleToggleLock,
      toggleVisibility: handleToggleVisibility,
      transform: handleTransform,
    }),
    [
      convert,
      getActionLabel,
      handleBooleanRaster,
      handleCopyTo,
      handleCopyToClipboard,
      handleCropToBbox,
      handleDelete,
      handleDuplicate,
      handleExtractMaskedArea,
      handleFitToBbox,
      handleLayerConfigAction,
      handleMerge,
      handleOpenProperties,
      handleRasterize,
      handleSaveToAssets,
      handleToggleLock,
      handleToggleVisibility,
      handleTransform,
      openRename,
      openRunWorkflow,
      startSelectObject,
      startFilter,
      reorder,
    ]
  );

  const runAction = useCallback(
    (action: LayerContextAction) => {
      void Promise.resolve()
        .then(() => action.handler({ ...actionState, effects }))
        .catch((error: unknown) => {
          notify.error(t('widgets.layers.actions.actionFailed'), toErrorMessage(error));
        });
    },
    [actionState, effects, notify, t]
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
              {menuLayout.map((section) => (
                <LayerMenuSection key={section.id} runAction={runAction} section={section} t={t} />
              ))}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <RenameDialog
        initialName={layer.name}
        isOpen={dialogKind === 'rename'}
        label={t('widgets.layers.actions.rename')}
        submitLabel={t('widgets.layers.actions.rename')}
        title={t('widgets.layers.actions.rename')}
        onClose={closeDialog}
        onSubmit={submitRename}
      />
      {dialogKind === 'run-workflow' ? (
        <RunLayerWorkflowDialog
          availability={workflowAvailability}
          engine={engine}
          isOpen
          layerId={layer.id}
          onClose={closeDialog}
        />
      ) : null}
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
 * menu's sibling-dialog state.
 *
 * Choosing a sibling-dialog action closes the menu, which nulls `target`. The
 * wrapper therefore owns the dialog-in-flight state and keeps rendering against
 * the last-known (sticky) target until the dialog closes (F1).
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
  // The layer a pending sibling dialog is anchored to. Captured while the live
  // target still exists, then retained until the dialog closes.
  const [dialogState, setDialogState] = useState<LayerMenuDialogState | null>(null);
  const renderTarget = resolveMenuTargetForRender(target, dialogState);

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
  const handleDialogKindChange = useCallback(
    (kind: LayerMenuDialogKind | null) => {
      setDialogState(kind && target ? { kind, target } : null);
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
      dialogKind={dialogState?.kind ?? null}
      engine={engine}
      index={index}
      layer={layer}
      layers={layers}
      lazyMount
      // The menu itself is visible only while the live target is set; once a
      // sibling dialog closes it (target → null), the subtree stays mounted.
      open={!!target}
      positioning={positioning}
      unmountOnExit
      onOpenChange={handleOpenChange}
      onDialogKindChange={handleDialogKindChange}
    />
  );
};

const stopPropagation = (event: { stopPropagation: () => void }): void => event.stopPropagation();

const SUBMENU_META: Record<LayerContextSubmenuId, { defaultLabel: string; icon: LucideIcon; labelKey: string }> = {
  'add-modifiers': {
    defaultLabel: 'Add modifiers',
    icon: PlusIcon,
    labelKey: 'widgets.layers.menu.addModifiers',
  },
  'add-regional': { defaultLabel: 'Add', icon: PlusIcon, labelKey: 'widgets.layers.menu.add' },
  arrange: { defaultLabel: 'Arrange', icon: ArrowUpDownIcon, labelKey: 'widgets.layers.menu.arrange' },
  boolean: { defaultLabel: 'Boolean operations', icon: MergeIcon, labelKey: 'widgets.layers.menu.booleanOperations' },
  'convert-to': { defaultLabel: 'Convert to', icon: ArrowRightLeftIcon, labelKey: 'widgets.layers.menu.convertTo' },
  'copy-to': { defaultLabel: 'Copy to', icon: CopyIcon, labelKey: 'widgets.layers.menu.copyTo' },
};

const SUBMENU_POSITIONING = { placement: 'right-start' } as const;

const LayerMenuSection = ({
  runAction,
  section,
  t,
}: {
  runAction: (action: LayerContextAction) => void;
  section: LayerContextMenuSection;
  t: (key: string, options: { defaultValue: string }) => string;
}) => {
  const content = section.items.map((item) => (
    <LayerMenuLayoutItem
      key={item.kind === 'action' ? item.action.id : item.id}
      compact={section.presentation === 'row'}
      item={item}
      runAction={runAction}
      t={t}
    />
  ));

  if (section.presentation === 'row') {
    return <HStack gap="1">{content}</HStack>;
  }

  return (
    <>
      <Menu.Separator borderColor="border.subtle" />
      {content}
    </>
  );
};

const LayerMenuLayoutItem = ({
  compact,
  item,
  runAction,
  t,
}: {
  compact: boolean;
  item: LayerContextMenuItem;
  runAction: (action: LayerContextAction) => void;
  t: (key: string, options: { defaultValue: string }) => string;
}) => {
  if (item.kind === 'action') {
    return compact ? (
      <LayerMenuIconActionItem action={item.action} runAction={runAction} t={t} />
    ) : (
      <LayerMenuActionItem action={item.action} runAction={runAction} t={t} />
    );
  }

  return <LayerMenuSubmenu compact={compact} item={item} runAction={runAction} t={t} />;
};

const LayerMenuSubmenu = ({
  compact,
  item,
  runAction,
  t,
}: {
  compact: boolean;
  item: Extract<LayerContextMenuItem, { kind: 'submenu' }>;
  runAction: (action: LayerContextAction) => void;
  t: (key: string, options: { defaultValue: string }) => string;
}) => {
  const meta = SUBMENU_META[item.id];
  const label = t(meta.labelKey, { defaultValue: meta.defaultLabel });

  return (
    <Menu.Root positioning={SUBMENU_POSITIONING}>
      <Menu.TriggerItem
        aria-label={label}
        flex={compact ? '1' : undefined}
        justifyContent={compact ? 'center' : undefined}
      >
        {compact ? (
          <Icon as={meta.icon} boxSize="4" color="fg.subtle" />
        ) : (
          <HStack gap="2" minW="0" w="full">
            <Icon as={meta.icon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
            <Text flex="1" fontSize="xs">
              {label}
            </Text>
            <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
          </HStack>
        )}
      </Menu.TriggerItem>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="13rem" py="1">
            {item.actions.map((action) => (
              <LayerMenuActionItem key={action.id} action={action} runAction={runAction} t={t} />
            ))}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const LayerMenuIconActionItem = ({
  action,
  runAction,
  t,
}: {
  action: LayerContextAction;
  runAction: (action: LayerContextAction) => void;
  t: (key: string, options: { defaultValue: string }) => string;
}) => {
  const onSelect = useCallback(() => runAction(action), [action, runAction]);

  return (
    <LayerMenuIconItem
      disabled={action.isDisabled}
      icon={action.icon}
      label={t(action.labelKey, { defaultValue: action.defaultLabel })}
      value={action.id}
      onSelect={onSelect}
    />
  );
};

const LayerMenuActionItem = ({
  action,
  runAction,
  t,
}: {
  action: LayerContextAction;
  runAction: (action: LayerContextAction) => void;
  t: (key: string, options: { defaultValue: string }) => string;
}) => {
  const onSelect = useCallback(() => runAction(action), [action, runAction]);

  return (
    <LayerMenuItem
      color={action.tone === 'danger' ? 'fg.error' : undefined}
      disabled={action.isDisabled}
      icon={action.icon}
      label={t(action.labelKey, { defaultValue: action.defaultLabel })}
      value={action.id}
      onSelect={onSelect}
    />
  );
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

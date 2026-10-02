import type {
  CanvasAdjustmentEntry,
  CanvasColorLabel,
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasMaskContract,
  CanvasMaskDenoiseContract,
  CanvasMaskNoiseContract,
  BooleanRasterOperation,
  LayerStackMoveKind,
  RegionalGuidanceReferenceImage,
} from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { LucideIcon } from 'lucide-react';
import type { ComponentProps, Dispatch, ReactNode } from 'react';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { galleryTransfers } from '@features/gallery';
import { useModelsSelector } from '@features/models';
import { IconButton, MenuActionItem, MenuContent, MenuIconItem, RenameDialog, Tooltip } from '@platform/ui';
import {
  canMergeSelectedRasters,
  getDocumentIndex,
  getDocumentLayer,
  getSourceContentRect,
  isGroupNode,
  isHideableLayer,
  isNodeHidden,
  isOverlayStack,
  lookupDocumentNodeState,
  renderableSourceOf,
} from '@workbench/canvas-engine/api';
import { getCanvasOperations } from '@workbench/canvas-operations/api';
import { focusOpenedWidget } from '@workbench/focusRegions';
import { formatHotkeyForPlatform } from '@workbench/hotkeys/keys';
import { publishLayerPanelSelection, readLayerPanelState, useLayerPanelState } from '@workbench/layerPanelState';
import { useNotify } from '@workbench/useNotify';
import { isCanvasInteractionLocked } from '@workbench/widgets/canvas/canvasInteractionLock';
import { useCanvasDocumentEditingLocked, useCanvasEngineRead } from '@workbench/widgets/canvas/engineStoreHooks';
import { reportLayerOperation, usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectId, useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import {
  ArrowRightLeftIcon,
  ArrowUpDownIcon,
  ChevronRightIcon,
  CopyIcon,
  MergeIcon,
  MoreVerticalIcon,
  PaletteIcon,
  SlidersHorizontalIcon,
} from 'lucide-react';
import { Fragment, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

export type LayerContextMenuEngine = Pick<
  CanvasEngineHandle,
  'document' | 'exports' | 'interaction' | 'layers' | 'projectId' | 'tools'
>;

import type {
  LayerContextMenuItem,
  LayerContextMenuRenderEntry,
  LayerContextMenuSection,
  LayerContextSubmenuId,
} from './layerContextMenuLayout';
import type { LayerMenuDialogKind, LayerMenuDialogState } from './layerMenuState';

import { resolveDefaultControlModelForBase } from './controlModelOptions';
import {
  actionTargets,
  getLayerContextActions,
  type LayerConfigPatchKind,
  type LayerContextAction,
  type LayerContextActionEffects,
  type LayerContextActionId,
  type LayerContextActionState,
  type LayerType,
} from './layerContextActions';
import {
  getLayerContextMenuGroupLayout,
  getLayerContextMenuLayerLabelKey,
  getLayerContextMenuLayout,
  getLayerContextMenuRenderEntries,
} from './layerContextMenuLayout';
import { copyLayerToClipboard, saveLayerToAssets } from './layerExportActions';
import { canGroupSelection, groupLayers } from './layerGroupCommands';
import { resolveMenuTargetForRender } from './layerMenuState';
import {
  DEFAULT_INPAINT_MASK_FILL,
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
  createIdentityAdjustment,
  createRegionalReferenceImage,
  fitLayerTransformToBbox,
  getControlTransparencyEffectPatch,
  getRegionalGuidanceAutoNegativePatch,
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
      referenceImages?: RegionalGuidanceReferenceImage[];
    }
  | {
      layerType: 'inpaint_mask';
      noise?: CanvasMaskNoiseContract | null;
      denoise?: CanvasMaskDenoiseContract | null;
    };

const PANEL_POSITIONING: MenuPositioning = { placement: 'bottom-end' };

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

const assertNever = (value: never): never => {
  throw new Error(`Unhandled layer action result: ${String(value)}`);
};

type LayerActionErrorStatus =
  | 'aborted'
  | 'busy'
  | 'disabled'
  | 'empty'
  | 'failed'
  | 'locked'
  | 'missing'
  | 'not-ready'
  | 'over-budget'
  | 'unsupported';

const LAYER_ACTION_ERROR_KEYS: Record<LayerActionErrorStatus, string> = {
  aborted: 'widgets.layers.actions.notReady',
  busy: 'widgets.layers.actions.busy',
  disabled: 'widgets.layers.actions.disabled',
  empty: 'widgets.layers.actions.empty',
  failed: 'widgets.layers.actions.operationFailed',
  locked: 'widgets.layers.actions.locked',
  missing: 'widgets.layers.actions.missing',
  'not-ready': 'widgets.layers.actions.notReady',
  'over-budget': 'widgets.layers.actions.overBudget',
  unsupported: 'widgets.layers.actions.unsupported',
};

const hasPureExportableLayerContent = (layer: CanvasLayerContract, document: CanvasDocumentContractV3): boolean => {
  if (!renderableSourceOf(layer)) {
    return false;
  }
  const contentRect = getSourceContentRect(layer, document);
  return contentRect.width > 0 && contentRect.height > 0;
};

interface LayerMenuProps {
  dispatch: Dispatch<CanvasProjectMutation>;
  engine: LayerContextMenuEngine | null;
  layer: CanvasLayerContract;
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
  /** Canvas-only items composed immediately before the terminal danger section. */
  beforeDangerItems?: ReactNode;
  /** Adds the legacy layer and Canvas group labels on the canvas surface. */
  showGroupLabels?: boolean;
}

/**
 * Share per-layer actions across panel and canvas menus. Arrange within type groups; merge uses global adjacency.
 * Keep dialogs outside the closing menu portal.
 */
const LayerMenu = ({
  dispatch,
  engine,
  layer,
  positioning,
  withTrigger,
  open,
  onOpenChange,
  lazyMount,
  unmountOnExit,
  dialogKind: controlledDialogKind,
  onDialogKindChange,
  beforeDangerItems,
  showGroupLabels,
}: LayerMenuProps) => {
  const commitPrepared = usePreparedCommit(engine);
  const { t } = useTranslation();
  const projectId = useActiveProjectId();
  const { widgets } = useWorkbenchCommands();
  const notify = useNotify();
  const base = useSelectedModelBase();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const defaultControlModel = useMemo(() => resolveDefaultControlModelForBase(models, base), [base, models]);
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
  const { selectedIds } = useLayerPanelState(projectId, document.selectedLayerId);
  const interactionLocked = isCanvasInteractionLocked(canvas, queueItems) || documentEditingLocked;
  const hasLiveContent = useCanvasEngineRead(
    engine,
    () => engine?.exports.hasExportableLayerContent(layer.id) ?? false
  );
  const hasSupportedContent = engine ? hasLiveContent : hasPureExportableLayerContent(layer, document);
  const [internalDialogKind, setInternalDialogKind] = useState<LayerMenuDialogKind | null>(null);
  const dialogKind = controlledDialogKind !== undefined ? controlledDialogKind : internalDialogKind;
  const setDialogKind = useCallback(
    (next: LayerMenuDialogKind | null) => {
      setInternalDialogKind(next);
      onDialogKindChange?.(next);
    },
    [onDialogKindChange]
  );

  const patchBase = useCallback(
    (label: string, forward: Partial<CanvasLayerContract>) => {
      commitPrepared(label, (model) => model.prepare({ id: layer.id, patch: forward, type: 'patch' }));
    },
    [commitPrepared, layer.id]
  );

  const patchConfig = useCallback(
    (label: string, forward: LayerConfigPatch) => {
      commitPrepared(label, (model) => model.prepare({ config: forward, id: layer.id, type: 'patch-config' }));
    },
    [commitPrepared, layer.id]
  );

  const reorder = useCallback(
    (kind: LayerStackMoveKind, label: string) => {
      commitPrepared(label, (model) =>
        model.prepare({ ids: actionTargets({ layer, selectedIds }), kind, type: 'move' })
      );
    },
    [commitPrepared, layer, selectedIds]
  );

  const targets = actionTargets({ layer, selectedIds });
  const canGroup = useCanvasEngineRead(engine, () => canGroupSelection(engine?.document.model() ?? null, targets));
  const canMerge = useCanvasEngineRead(engine, () => {
    const model = engine?.document.model() ?? null;
    return (
      !!engine &&
      !!model &&
      targets.length > 1 &&
      canMergeSelectedRasters(model.document, model.compileLeaves(), new Set(targets), (layerId) =>
        engine.exports.hasExportableLayerContent(layerId)
      )
    );
  });
  const canDeleteSelection = useCanvasEngineRead(
    engine,
    () => engine?.document.model()?.refusalFor({ ids: targets, type: 'remove' }) === null
  );
  const hiddenByAncestor =
    (lookupDocumentNodeState(document, layer.id)?.documentHidden ?? false) && !isNodeHidden(layer);
  const actionState = useMemo<LayerContextActionState>(
    () => ({
      canDeleteSelection,
      canGroupSelection: canGroup,
      canMergeSelection: canMerge,
      hiddenByAncestor,
      canRunWorkflow: workflowAvailability.canRunWorkflow,
      document,
      hasEngine: engine !== null,
      hasSupportedContent,
      hasWorkflowBindings: workflowAvailability.hasWorkflowBindings,
      interactionLocked,
      layer,
      modelBase: base,
      selectedIds,
    }),
    [
      base,
      hiddenByAncestor,
      canDeleteSelection,
      canGroup,
      canMerge,
      document,
      engine,
      hasSupportedContent,
      interactionLocked,
      layer,
      selectedIds,
      workflowAvailability.canRunWorkflow,
      workflowAvailability.hasWorkflowBindings,
    ]
  );
  const actions = useMemo(() => getLayerContextActions(actionState), [actionState]);
  const menuLayout = useMemo(() => getLayerContextMenuLayout(actions), [actions]);

  const getActionLabel = useCallback(
    (id: LayerContextActionId) => {
      const action = actions.find((entry) => entry.id === id);
      return action ? t(action.labelKey, { count: action.labelCount, defaultValue: action.defaultLabel }) : id;
    },
    [actions, t]
  );

  const makeStatusError = useCallback(
    (status: LayerActionErrorStatus): Error => new Error(t(LAYER_ACTION_ERROR_KEYS[status])),
    [t]
  );

  const handleDuplicate = useCallback(async () => {
    if (engine) {
      const panelSelection = readLayerPanelState(projectId, document.selectedLayerId);
      const sourceIds = panelSelection.selectedIds.includes(layer.id) ? panelSelection.selectedIds : [layer.id];
      try {
        const result = await engine.layers.duplicateLayers(sourceIds);
        if (result.status === 'duplicated') {
          publishLayerPanelSelection({
            primaryId: result.selectedLayerId,
            projectId,
            selectedIds: result.duplicateIds,
          });
          return;
        }
        if (result.status === 'busy') {
          return;
        }
        if (result.status !== 'nothing') {
          reportLayerOperation(result.status, notify.error, t);
          return;
        }
      } catch {
        // Report rejected transactions in the menu; the document remains unchanged.
      }
      notify.error(t('widgets.layers.actions.actionFailed'), t('widgets.layers.actions.copyFailed'));
      return;
    }
    commitPrepared(t('widgets.layers.actions.duplicate'), (model) =>
      model.prepare({ createId: createLayerId, ids: [layer.id], type: 'duplicate' })
    );
  }, [commitPrepared, document.selectedLayerId, engine, layer.id, notify, projectId, t]);

  const handleDelete = useCallback(() => {
    const ids = actionTargets({ layer, selectedIds });
    commitPrepared(getActionLabel('delete'), (model) => model.prepare({ ids, type: 'remove' }));
  }, [commitPrepared, getActionLabel, layer, selectedIds]);

  const handleGroup = useCallback(() => {
    const ids = actionTargets({ layer, selectedIds });
    const outcome = groupLayers(engine, projectId, ids, t('widgets.layers.actions.group'));
    if (outcome.status === 'refused') {
      throw makeStatusError(outcome.refusal.status === 'locked' ? 'locked' : 'unsupported');
    }
  }, [engine, layer, makeStatusError, projectId, selectedIds, t]);

  const handleMergeSelected = useCallback(() => {
    if (!engine) {
      return;
    }
    void engine.layers.mergeSelectedRasterLayers(actionTargets({ layer, selectedIds })).then((result) => {
      if (result === 'not-ready') {
        notify.error(t('widgets.layers.actions.actionFailed'), t('widgets.layers.groupActions.mergeNotReady'));
      } else if (result === 'over-budget') {
        notify.error(t('widgets.layers.actions.actionFailed'), t('widgets.layers.groupActions.mergeOverBudget'));
      } else if (result === 'failed') {
        notify.error(t('widgets.layers.actions.actionFailed'), t('widgets.layers.actions.operationFailed'));
      }
    });
  }, [engine, layer, notify, selectedIds, t]);

  const handleMerge = useCallback(() => {
    if (!engine) {
      throw makeStatusError('not-ready');
    }
    const result = engine.layers.mergeLayerDown(layer.id);
    if (result !== 'merged') {
      throw makeStatusError(result);
    }
  }, [engine, layer.id, makeStatusError]);

  const handleRasterize = useCallback(() => {
    if (!engine) {
      throw makeStatusError('not-ready');
    }
    const result = engine.layers.rasterizeLayer(layer.id);
    if (result !== 'rasterized') {
      throw makeStatusError(result);
    }
  }, [engine, layer.id, makeStatusError]);

  const addCopy = useCallback(
    (copied: CanvasLayerContract | null, label: string) => {
      if (!copied) {
        throw new Error(t('widgets.layers.actions.copyFailed'));
      }
      if (
        !engine?.layers.commitLayerCopy(
          label,
          layer.id,
          copied,
          engine.document.captureInsertionAnchor(copied.type, layer.id)
        )
      ) {
        throw new Error(t('widgets.layers.actions.copyFailed'));
      }
    },
    [engine, layer.id, t]
  );

  const convert = useCallback(
    (targetType: CanvasLayerContract['type'], label: string) => {
      const converted =
        layer.type === 'raster' && targetType === 'control'
          ? convertRasterToControl(layer, base, defaultControlModel)
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
      // Pass the immutable live object: the engine rejects stale menu actions
      // by identity and clones the inverse contract internally.
      if (!engine?.layers.commitLayerConversion(label, layer, converted)) {
        throw makeStatusError('not-ready');
      }
    },
    [base, defaultControlModel, engine, layer, makeStatusError]
  );

  const handleToggleVisibility = useCallback(() => {
    const targets = actionTargets({ layer, selectedIds });
    if (targets.length > 1) {
      const index = getDocumentIndex(document);
      const isEnabled = !targets.every((id) => index.byId.get(id)?.node.isEnabled ?? true);
      commitPrepared(
        t(isEnabled ? 'widgets.layers.actions.enableSelected' : 'widgets.layers.actions.disableSelected'),
        (model) => model.prepare({ type: 'set-enabled', updates: targets.map((id) => ({ id, isEnabled })) })
      );
      return;
    }
    patchBase(t('widgets.layers.actions.toggleVisibility'), { isEnabled: !layer.isEnabled });
  }, [commitPrepared, document, layer, patchBase, selectedIds, t]);

  const handleToggleHidden = useCallback(() => {
    const targets = actionTargets({ layer, selectedIds });
    if (targets.length > 1) {
      const index = getDocumentIndex(document);
      const nodes = targets.flatMap((id) => {
        const entry = index.byId.get(id);
        if (!entry) {
          return [];
        }
        // Mirrors the model: overlay-stack groups can hide; leaves ask isHideableLayer.
        return (isGroupNode(entry.node) ? isOverlayStack(entry.stack) : isHideableLayer(entry.node))
          ? [entry.node]
          : [];
      });
      const isHidden = !nodes.every((node) => isNodeHidden(node));
      commitPrepared(
        t(isHidden ? 'widgets.layers.actions.hideSelectedLayers' : 'widgets.layers.actions.showSelectedLayers'),
        (model) => model.prepare({ type: 'set-hidden', updates: nodes.map((node) => ({ id: node.id, isHidden })) })
      );
      return;
    }
    commitPrepared(t('widgets.layers.actions.toggleHidden'), (model) =>
      model.prepare({ type: 'set-hidden', updates: [{ id: layer.id, isHidden: !isNodeHidden(layer) }] })
    );
  }, [commitPrepared, document, layer, selectedIds, t]);

  const handleToggleLock = useCallback(() => {
    const targets = actionTargets({ layer, selectedIds });
    if (targets.length > 1) {
      const index = getDocumentIndex(document);
      const isLocked = !targets.every((id) => index.byId.get(id)?.node.isLocked ?? false);
      commitPrepared(
        t(isLocked ? 'widgets.layers.actions.lockSelected' : 'widgets.layers.actions.unlockSelected'),
        (model) => model.prepare({ type: 'set-locked', updates: targets.map((id) => ({ id, isLocked })) })
      );
      return;
    }
    patchBase(t('widgets.layers.actions.toggleLock'), { isLocked: !layer.isLocked });
  }, [commitPrepared, document, layer, patchBase, selectedIds, t]);

  const handleSetColorLabel = useCallback(
    (label: CanvasColorLabel | null) => {
      // `undefined` clears: the reducer writes the key as undefined and
      // serialization drops it, so "no label" round-trips as absence.
      patchBase(t('widgets.layers.menu.colorLabel'), { colorLabel: label ?? undefined });
    },
    [patchBase, t]
  );

  const openRename = useCallback(() => setDialogKind('rename'), [setDialogKind]);
  const closeDialog = useCallback(() => setDialogKind(null), [setDialogKind]);
  const openRunWorkflow = useCallback(() => setDialogKind('run-workflow'), [setDialogKind]);
  // Operations present their controls in the Properties pane, so starting one must bring that pane into view.
  const revealProperties = useCallback(
    (layerId: string) => {
      widgets.open({ region: 'right', widgetId: 'layers' });
      requestLayerProperties(layerId);
    },
    [widgets]
  );
  const startSelectObject = useCallback(
    (layerId: string) => {
      if (!engine) {
        throw makeStatusError('not-ready');
      }
      const result = getCanvasOperations(engine).startSelectObject(layerId);
      if (result !== 'started') {
        throw makeStatusError(result);
      }
      revealProperties(layerId);
    },
    [engine, makeStatusError, revealProperties]
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
      revealProperties(layerId);
    },
    [engine, makeStatusError, revealProperties]
  );
  const submitRename = useCallback(
    (name: string) => {
      patchBase(t('widgets.layers.actions.rename'), { name });
    },
    [patchBase, t]
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
    patchBase(getActionLabel('fit-to-bbox'), { transform });
  }, [bbox, documentRect, getActionLabel, layer, makeStatusError, patchBase]);

  const handleSaveToAssets = useCallback(async () => {
    if (!engine) {
      throw makeStatusError('not-ready');
    }

    const result = await saveLayerToAssets(layer.id, {
      exportLayer: engine.exports.exportBakedLayerBlob,
      upload: galleryTransfers.upload,
    });
    if (result !== 'saved' && result !== 'stale') {
      throw makeStatusError(result);
    }
  }, [engine, layer.id, makeStatusError]);

  const handleCopyToClipboard = useCallback(async () => {
    if (!engine) {
      throw makeStatusError('not-ready');
    }
    const status = await copyLayerToClipboard(layer.id, { exportLayer: engine.exports.exportBakedLayerBlob });
    if (status !== 'ok') {
      throw makeStatusError(status);
    }
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
      case 'over-budget':
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

  const handleOpenProperties = useCallback(() => {
    revealProperties(layer.id);
    focusOpenedWidget('right', 'layers');
  }, [layer.id, revealProperties]);

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
    const result = await engine.layers.copyLayerToRaster(layer.id);
    if (result.status !== 'copied') {
      throw makeStatusError(result.status);
    }
  }, [addCopy, engine, getActionLabel, layer, makeStatusError]);

  const handleCopyToControl = useCallback(() => {
    if (layer.type === 'raster') {
      addCopy(
        copyRasterToControl(layer, createLayerId(), base, defaultControlModel),
        getActionLabel('copy-to-control')
      );
    }
  }, [addCopy, base, defaultControlModel, getActionLabel, layer]);

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
        patchConfig(getActionLabel(id), getControlTransparencyEffectPatch(layer));
      } else if (id === 'regional-auto-negative' && layer.type === 'regional_guidance') {
        patchConfig(getActionLabel(id), getRegionalGuidanceAutoNegativePatch(layer));
      }
    },
    [getActionLabel, layer, patchConfig]
  );

  const handleAddMaskModifier = useCallback(
    (field: 'noise' | 'denoise') => {
      if (layer.type !== 'inpaint_mask') {
        return;
      }
      // A modifier that appeared since the menu rendered must not be stomped by the default.
      if (layer[field] !== undefined) {
        return;
      }
      // Legacy defaults: noise starts at 25%, the denoise limit at 80%.
      const value = field === 'noise' ? { isEnabled: true, level: 0.25 } : { isEnabled: true, limit: 0.8 };
      commitPrepared(
        t(field === 'noise' ? 'widgets.layers.actions.addNoise' : 'widgets.layers.actions.addDenoiseLimit'),
        (model) =>
          model.prepare({
            before: { [field]: null, layerType: 'inpaint_mask' },
            config: { [field]: value, layerType: 'inpaint_mask' },
            id: layer.id,
            type: 'patch-config',
          })
      );
    },
    [commitPrepared, layer, t]
  );

  const handleAddLayerRegion = useCallback(() => {
    if (layer.type !== 'raster' || layer.inpaint) {
      return;
    }
    commitPrepared(t('widgets.layers.actions.addRegenerateRegion'), (model) =>
      model.prepare({
        before: { inpaint: null, layerType: 'raster' },
        config: {
          inpaint: { fill: { ...DEFAULT_INPAINT_MASK_FILL }, isEnabled: true },
          layerType: 'raster',
        },
        id: layer.id,
        type: 'patch-config',
      })
    );
  }, [commitPrepared, layer, t]);

  const handleAddAdjustment = useCallback(
    (type: CanvasAdjustmentEntry['type']) => {
      if (layer.type !== 'raster') {
        return;
      }
      const entry = createIdentityAdjustment(type);
      const before = layer.adjustments ?? [];
      commitPrepared(t('widgets.layers.menu.addAdjustment'), (model) =>
        model.prepare({
          before: { adjustments: [...before], layerType: 'raster' },
          config: { adjustments: [...before, entry], layerType: 'raster' },
          id: layer.id,
          type: 'patch-config',
        })
      );
    },
    [commitPrepared, layer, t]
  );

  const handleAddReferenceImage = useCallback(() => {
    if (layer.type !== 'regional_guidance') {
      return;
    }
    commitPrepared(t('widgets.layers.regionalGuidance.referenceImages'), (model) =>
      model.prepare({
        before: { layerType: 'regional_guidance', referenceImages: [...layer.referenceImages] },
        config: {
          layerType: 'regional_guidance',
          referenceImages: [...layer.referenceImages, createRegionalReferenceImage(base)],
        },
        id: layer.id,
        type: 'patch-config',
      })
    );
  }, [base, commitPrepared, layer, t]);

  const effects = useMemo<LayerContextActionEffects>(
    () => ({
      addAdjustment: handleAddAdjustment,
      addLayerRegion: handleAddLayerRegion,
      addMaskModifier: handleAddMaskModifier,
      addReferenceImage: handleAddReferenceImage,
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
      group: handleGroup,
      fitToBbox: handleFitToBbox,
      mergeDown: handleMerge,
      mergeSelected: handleMergeSelected,
      openProperties: handleOpenProperties,
      openRename,
      openRunWorkflow,
      startSelectObject,
      startFilter,
      patchConfig: handleLayerConfigAction,
      rasterize: handleRasterize,
      reorder: (kind, actionId) => reorder(kind, getActionLabel(actionId)),
      saveToAssets: handleSaveToAssets,
      setColorLabel: handleSetColorLabel,
      toggleLock: handleToggleLock,
      toggleHidden: handleToggleHidden,
      toggleVisibility: handleToggleVisibility,
      transform: handleTransform,
    }),
    [
      convert,
      getActionLabel,
      handleAddAdjustment,
      handleAddLayerRegion,
      handleAddMaskModifier,
      handleAddReferenceImage,
      handleBooleanRaster,
      handleCopyTo,
      handleCopyToClipboard,
      handleCropToBbox,
      handleDelete,
      handleDuplicate,
      handleExtractMaskedArea,
      handleFitToBbox,
      handleGroup,
      handleMergeSelected,
      handleLayerConfigAction,
      handleMerge,
      handleOpenProperties,
      handleRasterize,
      handleSaveToAssets,
      handleSetColorLabel,
      handleToggleLock,
      handleToggleHidden,
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
  const menuRenderEntries = getLayerContextMenuRenderEntries(menuLayout, Boolean(beforeDangerItems));
  const groupLayout = showGroupLabels ? getLayerContextMenuGroupLayout(menuLayout, Boolean(beforeDangerItems)) : null;

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
              {groupLayout ? (
                <>
                  <Menu.ItemGroup>
                    <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                      {t(getLayerContextMenuLayerLabelKey(layer.type))}
                    </Menu.ItemGroupLabel>
                    {renderLayerMenuEntries({ entries: groupLayout.layerEntries, runAction, t })}
                  </Menu.ItemGroup>
                  {groupLayout.hasCanvasGroup ? (
                    <>
                      <Menu.Separator borderColor="border.subtle" />
                      <Menu.ItemGroup>
                        <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                          {t('widgets.labels.canvas')}
                        </Menu.ItemGroupLabel>
                        {beforeDangerItems}
                      </Menu.ItemGroup>
                    </>
                  ) : null}
                  {renderLayerMenuEntries({ entries: groupLayout.trailingEntries, runAction, t })}
                </>
              ) : (
                renderLayerMenuEntries({
                  beforeDangerItems,
                  entries: menuRenderEntries,
                  runAction,
                  t,
                })
              )}
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
  dispatch: Dispatch<CanvasProjectMutation>;
  engine: LayerContextMenuEngine | null;
  layer: CanvasLayerContract;
}

/** The layers-panel per-row context menu: a ⋯ trigger button, opened below it. */
export const LayerContextMenu = (props: LayerRowMenuProps) => (
  <LayerMenu {...props} positioning={PANEL_POSITIONING} showGroupLabels withTrigger />
);

/** The layer + pointer position a canvas right-click resolved to. */
export interface CanvasLayerContextMenuTarget {
  layerId: string;
  x: number;
  y: number;
}

/**
 * Anchor {@link LayerMenu} at the cursor for the supplied layer id, resolved against the live document. Keep a
 * sticky dialog target after menu closure so sibling dialogs survive; changing layer resets their state.
 */
export const CanvasLayerContextMenu = ({
  beforeDangerItems,
  dispatch,
  engine,
  target,
  showGroupLabels,
  onClose,
}: {
  beforeDangerItems?: ReactNode;
  dispatch: Dispatch<CanvasProjectMutation>;
  engine: LayerContextMenuEngine | null;
  target: CanvasLayerContextMenuTarget | null;
  showGroupLabels?: boolean;
  onClose: () => void;
}) => {
  // The layer a pending sibling dialog is anchored to. Captured while the live
  // target still exists, then retained until the dialog closes.
  const [dialogState, setDialogState] = useState<LayerMenuDialogState | null>(null);
  const renderTarget = resolveMenuTargetForRender(target, dialogState);

  const layerId = renderTarget?.layerId ?? null;
  const layer = useActiveProjectSelector((project) =>
    layerId ? (getDocumentLayer(project.canvas.document, layerId) ?? undefined) : undefined
  );

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
      beforeDangerItems={beforeDangerItems}
      dispatch={dispatch}
      dialogKind={dialogState?.kind ?? null}
      engine={engine}
      layer={layer}
      lazyMount
      // Hide the menu when target clears while preserving the sibling-dialog subtree.
      open={!!target}
      positioning={positioning}
      showGroupLabels={showGroupLabels}
      unmountOnExit
      onOpenChange={handleOpenChange}
      onDialogKindChange={handleDialogKindChange}
    />
  );
};

const stopPropagation = (event: { stopPropagation: () => void }): void => event.stopPropagation();

const SUBMENU_META: Record<LayerContextSubmenuId, { defaultLabel: string; icon: LucideIcon; labelKey: string }> = {
  'add-adjustment': {
    defaultLabel: 'Add adjustment',
    icon: SlidersHorizontalIcon,
    labelKey: 'widgets.layers.menu.addAdjustment',
  },
  arrange: { defaultLabel: 'Arrange', icon: ArrowUpDownIcon, labelKey: 'widgets.layers.menu.arrange' },
  boolean: { defaultLabel: 'Boolean operations', icon: MergeIcon, labelKey: 'widgets.layers.menu.booleanOperations' },
  'color-label': { defaultLabel: 'Color label', icon: PaletteIcon, labelKey: 'widgets.layers.menu.colorLabel' },
  'convert-to': { defaultLabel: 'Convert to', icon: ArrowRightLeftIcon, labelKey: 'widgets.layers.menu.convertTo' },
  'copy-to': { defaultLabel: 'Copy to', icon: CopyIcon, labelKey: 'widgets.layers.menu.copyTo' },
};

const SUBMENU_POSITIONING = { placement: 'right-start' } as const;
const QUICK_MENU_TOOLTIP_CONTENT_PROPS = { fontSize: '2xs' } as const;
const QUICK_MENU_TOOLTIP_POSITIONING_PROPS = { placement: 'top' } as const;

const renderLayerMenuEntries = ({
  beforeDangerItems,
  entries,
  runAction,
  t,
}: {
  beforeDangerItems?: ReactNode;
  entries: readonly LayerContextMenuRenderEntry[];
  runAction: (action: LayerContextAction) => void;
  t: (key: string, options?: { defaultValue: string }) => string;
}) =>
  entries.map((entry) =>
    entry.kind === 'slot' ? (
      <Fragment key={entry.id}>
        <Menu.Separator borderColor="border.subtle" />
        {beforeDangerItems}
      </Fragment>
    ) : (
      <LayerMenuSection key={entry.section.id} runAction={runAction} section={entry.section} t={t} />
    )
  );

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
          <Tooltip
            showArrow
            content={label}
            contentProps={QUICK_MENU_TOOLTIP_CONTENT_PROPS}
            openDelay={300}
            positioning={QUICK_MENU_TOOLTIP_POSITIONING_PROPS}
          >
            <Icon as={meta.icon} boxSize="4" color="fg" />
          </Tooltip>
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
  t: (key: string, options: { count?: number; defaultValue: string }) => string;
}) => {
  const onSelect = useCallback(() => runAction(action), [action, runAction]);

  return (
    <MenuIconItem
      disabled={action.isDisabled}
      icon={action.icon}
      label={t(action.labelKey, { count: action.labelCount, defaultValue: action.defaultLabel })}
      tone={action.tone}
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
  t: (key: string, options: { count?: number; defaultValue: string }) => string;
}) => {
  const onSelect = useCallback(() => runAction(action), [action, runAction]);

  return (
    <LayerMenuItem
      disabled={action.isDisabled}
      hint={action.hint}
      icon={action.icon}
      iconColor={action.iconColor}
      label={t(action.labelKey, { count: action.labelCount, defaultValue: action.defaultLabel })}
      tone={action.tone}
      value={action.id}
      onSelect={onSelect}
    />
  );
};

const LayerMenuItem = ({
  disabled,
  hint,
  icon,
  iconColor,
  label,
  onSelect,
  tone,
  value,
}: {
  disabled?: boolean;
  /** A raw hotkey string, formatted per platform and shown as trailing keycaps. */
  hint?: string;
  icon: LucideIcon;
  iconColor?: string;
  label: string;
  onSelect: () => void;
  tone?: 'danger';
  value: string;
}) => (
  <MenuActionItem
    disabled={disabled}
    hintParts={hint ? formatHotkeyForPlatform(hint) : undefined}
    icon={icon}
    iconColor={iconColor}
    label={label}
    tone={tone}
    value={value}
    onSelect={onSelect}
  />
);

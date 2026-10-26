import type { ArchitectureCapabilitiesSnapshot } from '@features/generation/runtime';
import type { ModelConfig } from '@features/models';
import type {
  CanvasControlAdapterContract,
  CanvasControlLayerContract,
  CanvasStructuralEngine,
  ControlType,
} from '@workbench/canvas-engine/api';
import type { ControlLayerIntent, ControlLayerIntentResult } from '@workbench/canvas-operations/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { HStack, Separator, Stack, Text } from '@chakra-ui/react';
import { getControlModelUnusableReason } from '@features/generation/graph';
import {
  ensureArchitectureCapabilitiesLoaded,
  getArchitectureCapabilitiesSnapshot,
  subscribeArchitectureCapabilities,
} from '@features/generation/runtime';
import { useCapabilities } from '@features/identity';
import { getModelBaseLabel, useModelsSelector, useStartersSelector } from '@features/models';
import { focusFirstOperable } from '@platform/react/focusIfUnclaimed';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Button } from '@platform/ui/Button';
import { useNavigate } from '@tanstack/react-router';
import { CONTROL_TYPES, getDocumentLeaves } from '@workbench/canvas-engine/api';
import { getCanvasOperations } from '@workbench/canvas-operations/api';
import { getControlImageState } from '@workbench/canvasLayerContent';
import { deriveControlKind } from '@workbench/controlAdapters';
import { createControlLayerCheckContext, describeControlLayer } from '@workbench/controlLayerChecks';
import {
  controlModelMemoryKey,
  getTypeAvailability,
  isRememberableControlPick,
} from '@workbench/controlModelResolution';
import { patchControlModelMemory, useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import {
  useCanvasDocumentEditingLocked,
  useCanvasEngineRead,
  useControlDetectionStatus,
} from '@workbench/widgets/canvas/engineStoreHooks';
import { baselineConfig, useStructuralPreview } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectId } from '@workbench/WorkbenchContext';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { ControlAdapterRows, type ControlAdapterEditing } from './ControlAdapterRows';
import { ControlDetectionRow } from './ControlDetectionRow';
import { getReidentifyPendingControlModels } from './controlModelOptions';
import { ControlModelRow } from './ControlModelRow';
import { deriveControlPanel } from './controlPanelModel';
import { ControlSourceRow } from './ControlSourceRow';
import { ControlTypeRow } from './ControlTypeRow';
import { isAutoControlLayerName, nextControlTypeLayerName } from './layerOps';
import { useControlIntentReporter, useControlSourceActions } from './useControlSourceActions';
import { useSelectedMainModel } from './useSelectedMainModel';

const selectCapabilitiesStatus = (snapshot: ArchitectureCapabilitiesSnapshot) => snapshot.status;
const selectCapabilitiesRevision = (snapshot: ArchitectureCapabilitiesSnapshot) => snapshot.revision;
const selectControlModelMemory = (preferences: { controlModelMemory: Record<string, string> }) =>
  preferences.controlModelMemory;
const selectStarterModels = (snapshot: {
  status: string;
  response: { starter_models: { source: string }[] } | null;
}) => (snapshot.status === 'loaded' ? (snapshot.response?.starter_models ?? null) : null);

/** Read from the engine in place of a reason when the layer does not contribute: its ambient warning applies. */
const AMBIENT = 'ambient';

/** Held back from an empty model list, so the caption can say why it is empty. */
const HIDDEN_MODEL_REASONS: ReadonlySet<string> = new Set(['lllite_inpaint_adapter', 'lllite_channels_unknown']);

export type ControlLayerSettingsEngine = CanvasStructuralEngine &
  Pick<CanvasEngineHandle, 'document' | 'exports' | 'interaction' | 'lifecycle' | 'projectId' | 'tools'>;

interface ControlLayerSettingsProps {
  engine: ControlLayerSettingsEngine | null;
  layer: CanvasControlLayerContract;
}

/**
 * The control layer Properties panel: Image · Type · Detection · Model, then Weight · Steps · Priority. The rows stay
 * put; only their values change. Type, detector and image edits go through the canvas operations, which detect the
 * map in the background; adapter values commit through canvas undo like any layer property.
 */
export const ControlLayerSettings = ({ engine, layer }: ControlLayerSettingsProps) => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const projectId = useActiveProjectId();
  const { canManageModels } = useCapabilities();
  const { cancel: cancelPreview, commit: commitPrepared, preview: previewStructural } = useStructuralPreview(engine);
  const editingLocked = useCanvasDocumentEditingLocked(engine);
  const models = useModelsSelector((snapshot) => snapshot.models);
  const mainModel = useSelectedMainModel();
  const baseId = mainModel?.base ?? null;
  const remembered = useWorkbenchPreferenceSelector(selectControlModelMemory);
  const starterModels = useStartersSelector(selectStarterModels);
  const starterSources = useMemo(
    () => (starterModels ? new Set(starterModels.map((model) => model.source)) : null),
    [starterModels]
  );
  const captionId = useId();

  const capabilitiesStatus = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    selectCapabilitiesStatus
  );
  // Kind support is read from the capability table when a check runs, so its arrival must re-run the checks.
  const capabilitiesRevision = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    selectCapabilitiesRevision
  );
  const checkContext = useMemo(() => {
    void capabilitiesRevision;
    return createControlLayerCheckContext(baseId, remembered);
  }, [baseId, capabilitiesRevision, remembered]);
  const described = useMemo(
    () => describeControlLayer(layer, { ...checkContext, mainModel, models }),
    [checkContext, layer, mainModel, models]
  );
  // Judged as Invoke judges it: contributing, content-bearing control layers above claim limited slots first.
  const sequenceReason = useCanvasEngineRead(engine, () => {
    if (!engine || !mainModel || !engine.exports.hasExportableLayerContent(layer.id)) {
      return AMBIENT;
    }
    const leaves = engine.document.model()?.compileLeaves() ?? [];
    const index = leaves.findIndex((leaf) => leaf.layer.id === layer.id);
    if (!(leaves[index]?.contributionEnabled ?? false)) {
      return AMBIENT;
    }
    const earlier = leaves
      .slice(0, Math.max(0, index))
      .filter((leaf) => leaf.contributionEnabled && engine.exports.hasExportableLayerContent(leaf.id))
      .map((leaf) => leaf.layer)
      .filter((candidate): candidate is CanvasControlLayerContract => candidate.type === 'control');
    return describeControlLayer(layer, { ...checkContext, earlier, mainModel, models }).reason;
  });
  const reason = sequenceReason === AMBIENT ? described.reason : (sequenceReason as typeof described.reason);
  const availability = useMemo(
    () => getTypeAvailability(baseId, models, checkContext.supportsKind),
    [baseId, checkContext, models]
  );
  const status = useControlDetectionStatus(engine, layer.id);
  const sourceHasPixels = useCanvasEngineRead(
    engine,
    () => engine?.exports.hasExportableLayerContent(layer.id) ?? false
  );
  const imageState = getControlImageState(layer, sourceHasPixels);
  const { resolution } = described;
  // Installed models of the base the list holds back: said in the caption when nothing else is left to pick.
  const hiddenModelReasons = useMemo(() => {
    if (baseId === null || resolution.model) {
      return [];
    }
    const reasons = models
      .filter((model) => model.base === baseId)
      .map((model) => {
        const kind = deriveControlKind(model);
        return kind ? getControlModelUnusableReason(model, kind) : null;
      })
      .filter((code): code is NonNullable<typeof code> => code !== null && HIDDEN_MODEL_REASONS.has(code));
    return [...new Set(reasons)].sort();
  }, [baseId, models, resolution.model]);
  const reidentifyPending = useMemo(
    () => (layer.controlType ? getReidentifyPendingControlModels(models, baseId) : []),
    [baseId, layer.controlType, models]
  );
  const base = baseId ? { id: baseId, label: getModelBaseLabel(baseId) } : null;
  const panel = deriveControlPanel({
    availability,
    base,
    hiddenModelReasons: reidentifyPending.length > 0 ? [] : hiddenModelReasons,
    imageState,
    layer,
    map: described.map,
    reason,
    resolution,
    status,
  });

  const reportIntent = useControlIntentReporter();
  const applyIntent = useCallback(
    (intent: ControlLayerIntent): ControlLayerIntentResult | null => {
      if (!engine) {
        return null;
      }
      const result = getCanvasOperations(engine).applyControlLayerIntent(layer.id, intent);
      reportIntent(result);
      return result;
    },
    [engine, layer.id, reportIntent]
  );
  const retryDetection = useCallback(() => {
    if (engine) {
      getCanvasOperations(engine).retryControlDetection(layer.id);
    }
  }, [engine, layer.id]);

  const source = useControlSourceActions({ applyIntent, engine, layerId: layer.id });
  const sourceLabel = (() => {
    if (imageState !== 'image') {
      return t('widgets.layers.control.source.drawn');
    }
    const pixels = layer.original ?? layer.source;
    if (pixels.type !== 'image') {
      return '';
    }
    const origin = source.origins[pixels.image.imageName];
    return origin?.kind === 'file'
      ? origin.name
      : origin?.kind === 'frame'
        ? t('widgets.layers.control.source.fromFrame')
        : t('widgets.layers.control.source.imageSize', { height: pixels.image.height, width: pixels.image.width });
  })();

  const handleTypeChange = useCallback(
    (controlType: ControlType, typeLabel: string) => {
      const typeLabels = CONTROL_TYPES.map((type) => t(`widgets.layers.control.types.${type}`));
      const document = engine?.document.getDocument();
      const names = document
        ? getDocumentLeaves(document)
            .filter((leaf) => leaf.id !== layer.id)
            .map((leaf) => leaf.name)
        : [];
      applyIntent({
        controlType,
        kind: 'set-type',
        name: isAutoControlLayerName(layer.name, typeLabels) ? nextControlTypeLayerName(typeLabel, names) : undefined,
      });
    },
    [applyIntent, engine, layer.id, layer.name, t]
  );
  const handleDetectorChange = useCallback(
    (detector: string | null) => applyIntent({ detector, kind: 'set-detector' }),
    [applyIntent]
  );
  const handleSettingsChange = useCallback(
    (settings: Record<string, unknown>) => applyIntent({ kind: 'set-detector-settings', settings }),
    [applyIntent]
  );

  // A previewed gesture records from where its preview started; any other change from the live adapter.
  const commitAdapter = useCallback(
    (next: Partial<CanvasControlAdapterContract>, label: string) => {
      const config = { adapter: next, layerType: 'control' } as const;
      commitPrepared(label, (model, baseline) =>
        model.prepare({ before: baselineConfig(baseline, config), config, id: layer.id, type: 'patch-config' })
      );
    },
    [commitPrepared, layer.id]
  );
  const adapterEditing = useMemo<ControlAdapterEditing>(
    () => ({
      cancel: cancelPreview,
      commit: commitAdapter,
      preview: (patch) => {
        previewStructural({
          config: { adapter: patch, layerType: 'control' },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        });
      },
    }),
    [cancelPreview, commitAdapter, layer.id, previewStructural]
  );

  const memoryKey = baseId && layer.controlType ? controlModelMemoryKey(baseId, layer.controlType) : null;
  // Account-wide and outside canvas undo, so only a pick that fits the type is remembered for later layers; any other
  // pick (purpose not set, another family) stays this layer's own.
  const handleModelPick = useCallback(
    (model: ModelConfig) => {
      const kind = deriveControlKind(model);
      commitAdapter(kind ? { kind, model: model.key } : { model: model.key }, t('widgets.layers.control.model'));
      if (memoryKey && isRememberableControlPick(resolution, model)) {
        void patchControlModelMemory(memoryKey, model.key);
      }
    },
    [commitAdapter, memoryKey, resolution, t]
  );
  // Reset undoes what drives the layer: the remembered pick is forgotten only when it is the model shown.
  const forgetsRemembered =
    memoryKey !== null && resolution.model !== null && remembered[memoryKey] === resolution.model.key;
  const handleModelReset = useCallback(() => {
    // With no model the stored kind is advisory (normalization keeps the values the resolved kind accepts).
    commitAdapter({ model: null }, t('widgets.layers.control.model'));
    if (forgetsRemembered && memoryKey) {
      void patchControlModelMemory(memoryKey, null);
    }
  }, [commitAdapter, forgetsRemembered, memoryKey, t]);
  const findModels = useCallback(() => {
    void navigate({ search: { project: projectId }, to: '/models' });
  }, [navigate, projectId]);

  const [hasRequestedCapabilitiesRetry, setHasRequestedCapabilitiesRetry] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const retryCapabilities = useCallback(() => {
    if (hasRequestedCapabilitiesRetry && capabilitiesStatus === 'loading') {
      return;
    }
    setHasRequestedCapabilitiesRetry(true);
    // The request belongs to this retry; a later reload started elsewhere is not this panel's retry.
    void ensureArchitectureCapabilitiesLoaded().then(() => setHasRequestedCapabilitiesRetry(false));
  }, [capabilitiesStatus, hasRequestedCapabilitiesRetry]);
  // Transfer retry-button focus to the loaded panel during ref cleanup before removal can send it to body.
  const handOverFocusOnLoad = useCallback((surface: HTMLDivElement | null) => {
    if (!surface) {
      return undefined;
    }
    return () => {
      if (surface.contains(document.activeElement) && getArchitectureCapabilitiesSnapshot().status === 'loaded') {
        focusFirstOperable(rootRef.current);
      }
    };
  }, []);
  const capabilitiesUnavailable = reason === 'capabilities_unavailable';
  // The click flips the status to `loading` synchronously; keeping the failure surface mounted for the
  // retry it started keeps the button, and the user's focus, in place.
  const isRetryingCapabilities = hasRequestedCapabilitiesRetry && capabilitiesStatus === 'loading';
  const showCapabilitiesFailure = capabilitiesUnavailable && (capabilitiesStatus === 'error' || isRetryingCapabilities);

  return (
    <Stack ref={rootRef} gap="1.5">
      <ControlSourceRow
        busy={source.busy}
        disabled={editingLocked || !engine}
        imageName={panel.image.imageName}
        label={sourceLabel}
        state={panel.image.state}
        onFrame={source.fromFrame}
        onDraw={source.draw}
        onUpload={source.upload}
      />
      <ControlTypeRow
        describedBy={captionId}
        disabled={editingLocked || !engine}
        model={panel.type}
        onChange={handleTypeChange}
      />
      <ControlDetectionRow
        disabled={editingLocked || !engine}
        model={panel.detection}
        onDetect={retryDetection}
        onDetectorChange={handleDetectorChange}
        onSettingsChange={handleSettingsChange}
      />
      <ControlModelRow
        canManageModels={canManageModels}
        captionId={captionId}
        controlType={layer.controlType ?? null}
        disabled={editingLocked}
        model={panel.model}
        reidentifyPending={reidentifyPending}
        rememberedHint={
          forgetsRemembered && layer.controlType && base
            ? t('widgets.layers.control.modelRememberedHint', {
                base: base.label,
                type: t(`widgets.layers.control.typeNouns.${layer.controlType}`),
              })
            : null
        }
        resolution={resolution}
        starterSources={starterSources}
        onFindModels={findModels}
        onPick={handleModelPick}
        onReset={handleModelReset}
      />
      <Separator borderColor="border.subtle" />
      <ControlAdapterRows
        adapter={layer.adapter}
        disabled={editingLocked}
        editing={adapterEditing}
        model={panel.adapter}
      />
      {showCapabilitiesFailure ? (
        <HStack ref={handOverFocusOnLoad} aria-busy={isRetryingCapabilities} gap="2" role="alert">
          <Text color="fg.warning" flex="1" fontSize="xs">
            {t('widgets.layers.control.capabilitiesLoadFailed')}
          </Text>
          {/* `aria-disabled` rather than `disabled`: a disabled button drops the focus it holds. */}
          <Button
            aria-busy={isRetryingCapabilities}
            aria-disabled={isRetryingCapabilities}
            variant="outline"
            onClick={retryCapabilities}
          >
            {t('common.retry')}
          </Button>
        </HStack>
      ) : null}
      {capabilitiesUnavailable && !showCapabilitiesFailure ? (
        <Text color="fg.muted" fontSize="xs" role="status">
          {t('widgets.layers.control.capabilitiesLoading')}
        </Text>
      ) : null}
    </Stack>
  );
};

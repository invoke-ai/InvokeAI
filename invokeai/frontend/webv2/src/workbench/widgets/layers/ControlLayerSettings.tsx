import type { SelectValueChangeDetails, SliderValueChangeDetails } from '@chakra-ui/react';
import type { ArchitectureCapabilitiesSnapshot } from '@features/generation/runtime';
import type {
  CanvasControlAdapterContract,
  CanvasControlLayerContract,
  CanvasDocumentCapability,
} from '@workbench/canvas-engine/api';
import type { LayerFilterOperationEngine } from '@workbench/widgets/layers/LayerFilterOperationButton';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';

import { createListCollection, HStack, Stack, Switch, Text } from '@chakra-ui/react';
import {
  CONTROL_ADAPTER_KINDS,
  getControlModelUnusableReason,
  getSuggestedControlKind,
  isControlKindSupportedForBase,
  type ControlAdapterKind,
} from '@features/generation/graph';
import {
  ensureArchitectureCapabilitiesLoaded,
  getArchitectureCapabilitiesSnapshot,
  subscribeArchitectureCapabilities,
} from '@features/generation/runtime';
import { useModelsSelector } from '@features/models';
import { focusFirstOperable } from '@platform/react/focusIfUnclaimed';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Button, Field, Select, Slider } from '@platform/ui';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { getCanvasOperations, resolveDefaultFilterForModel } from '@workbench/canvas-operations/api';
import { CONTROL_ADAPTER_DEFAULTS, CONTROL_KIND_BASE } from '@workbench/controlAdapters';
import { describeControlLayerReason, getControlLayerReasonInSequence } from '@workbench/controlLayerChecks';
import { useCanvasEngineRead } from '@workbench/widgets/canvas/engineStoreHooks';
import { useStructuralPreview } from '@workbench/widgets/canvas/useStructuralCommit';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { getCompatibleControlModels, switchControlAdapterKind } from './controlModelOptions';
import { LayerFilterOperationButton } from './LayerFilterOperationButton';
import { CONTROL_WEIGHT_BOUNDS } from './layerOps';
import { runLayerFilterOperation } from './layerPropertiesOperation';
import { useSelectedMainModel } from './useSelectedMainModel';

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: true } as const;

const selectCapabilitiesStatus = (snapshot: ArchitectureCapabilitiesSnapshot) => snapshot.status;
const selectCapabilitiesRevision = (snapshot: ArchitectureCapabilitiesSnapshot) => snapshot.revision;
const CONTROL_MODES: readonly NonNullable<CanvasControlAdapterContract['controlMode']>[] = [
  'balanced',
  'more_prompt',
  'more_control',
  'unbalanced',
];

const formatUnitPercent = (value: number): string => `${Math.round(value * 100)}%`;
const formatWeight = (value: number): string => value.toFixed(2);

/** Quiet warning tint for the model select while no control model is chosen. */
const MISSING_MODEL_VALUE_TEXT_PROPS = { color: 'fg.warning' } as const;

interface ControlLayerSettingsProps {
  engine:
    | (CanvasStructuralEngine & LayerFilterOperationEngine & { readonly document: CanvasDocumentCapability })
    | null;
  layer: CanvasControlLayerContract;
  onOperationStarted(): void;
}

/** Edit adapters through canvas undo; utility-queue filter previews leave the document untouched until Apply. */
export const ControlLayerSettings = ({ engine, layer, onOperationStarted }: ControlLayerSettingsProps) => {
  const { t } = useTranslation();
  const { cancel: cancelPreview, commit: commitPrepared, preview: previewStructural } = useStructuralPreview(engine);
  const models = useModelsSelector((snapshot) => snapshot.models);
  const mainModel = useSelectedMainModel();
  const base = mainModel?.base ?? null;
  const { adapter } = layer;
  const weightInputMin = adapter.kind === 'z_image_control' ? 0 : CONTROL_WEIGHT_BOUNDS.inputMin;

  const commitAdapter = useCallback(
    (next: Partial<CanvasControlAdapterContract>, before: Partial<CanvasControlAdapterContract>, label: string) => {
      commitPrepared(label, (model) =>
        model.prepare({
          before: { adapter: before, layerType: 'control' },
          config: { adapter: next, layerType: 'control' },
          id: layer.id,
          type: 'patch-config',
        })
      );
    },
    [commitPrepared, layer.id]
  );

  const capabilitiesStatus = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    selectCapabilitiesStatus
  );
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
  // Read supported adapter kinds inside the capability selector so successful retries replace the pre-load empty
  // list.
  const kindOptions = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(
      () =>
        CONTROL_ADAPTER_KINDS.filter((kind) =>
          // Without a main model there is no base to offer a base's own kind for.
          base ? isControlKindSupportedForBase(base, kind) : CONTROL_KIND_BASE[kind] === undefined
        ),
      [base]
    )
  );
  // The kind to offer when this layer's kind is one the main model cannot run.
  const suggestedKind = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(
      () => (base && !isControlKindSupportedForBase(base, adapter.kind) ? getSuggestedControlKind(base) : null),
      [adapter.kind, base]
    )
  );
  const kindCollection = useMemo(
    () =>
      createListCollection({
        items: kindOptions.map((kind) => ({ label: t(`widgets.layers.control.kinds.${kind}`), value: kind })),
      }),
    [kindOptions, t]
  );

  // Adapter models matching the current kind + base (mirrors the generate model list). A kind the main model cannot
  // run offers none: the fix is switching kind, and listing its models would present them as usable.
  const modelOptions = useMemo(
    () => (suggestedKind ? [] : getCompatibleControlModels(models, base, adapter.kind)),
    [adapter.kind, base, models, suggestedKind]
  );
  // Why an empty list is empty, when installed models of the right family were held back.
  const hiddenModelReasons = useMemo(() => {
    const kindBase = CONTROL_KIND_BASE[adapter.kind];
    if (modelOptions.length > 0 || !kindBase || base !== kindBase) {
      return [];
    }
    const reasons = models
      .filter((model) => model.base === base)
      .map((model) => getControlModelUnusableReason(model, adapter.kind))
      .filter((reason) => reason === 'lllite_inpaint_adapter' || reason === 'lllite_channels_unknown');
    return [...new Set(reasons)].sort();
  }, [adapter.kind, base, modelOptions.length, models]);
  const modelCollection = useMemo(
    () => createListCollection({ items: modelOptions.map((model) => ({ label: model.name, value: model.key })) }),
    [modelOptions]
  );

  const switchKind = useCallback(
    (kind: ControlAdapterKind) => {
      if (kind === adapter.kind) {
        return;
      }
      commitAdapter(
        switchControlAdapterKind(adapter, kind, models, base),
        { ...adapter, beginEndStepPct: [...adapter.beginEndStepPct] },
        t('widgets.layers.control.kind')
      );
    },
    [adapter, base, commitAdapter, models, t]
  );
  const handleKindChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const kind = value[0] as ControlAdapterKind | undefined;
      if (kind) {
        switchKind(kind);
      }
    },
    [switchKind]
  );
  const handleSwitchToSuggestedKind = useCallback(() => {
    if (suggestedKind) {
      switchKind(suggestedKind);
    }
  }, [suggestedKind, switchKind]);

  const handleModelChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const model = value[0] ?? null;
      if (model !== adapter.model) {
        commitAdapter({ model }, { model: adapter.model }, t('widgets.layers.control.model'));
        const selected = models.find((candidate) => candidate.key === model);
        const recommendation = resolveDefaultFilterForModel(selected);
        if (recommendation && !layer.filter) {
          runLayerFilterOperation(
            () => (engine ? getCanvasOperations(engine).startFilterOperation(layer.id, recommendation) : 'not-ready'),
            onOperationStarted
          );
        }
      }
    },
    [adapter.model, commitAdapter, engine, layer.filter, layer.id, models, onOperationStarted, t]
  );

  const handleControlModeChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const mode = value[0] as CanvasControlAdapterContract['controlMode'] | undefined;
      if (mode && mode !== adapter.controlMode) {
        commitAdapter({ controlMode: mode }, { controlMode: adapter.controlMode }, t('widgets.layers.control.mode'));
      }
    },
    [adapter.controlMode, commitAdapter, t]
  );

  // Where a weight gesture started; outside render closures because a drag keeps the handlers it started with.
  const weightBeforeRef = useRef<number | null>(null);
  const handleWeightChange = useCallback(
    (next: number) => {
      if (
        previewStructural({
          config: { adapter: { weight: next }, layerType: 'control' },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        })
      ) {
        weightBeforeRef.current ??= adapter.weight;
      }
    },
    [adapter.weight, previewStructural, layer.id]
  );
  const handleWeightChangeEnd = useCallback(
    (next: number) => {
      const before = weightBeforeRef.current;
      weightBeforeRef.current = null;
      if (before === null) {
        return;
      }
      if (next === before) {
        cancelPreview({
          config: { adapter: { weight: before }, layerType: 'control' },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        });
        return;
      }
      commitAdapter({ weight: next }, { weight: before }, t('widgets.layers.control.weight'));
    },
    [cancelPreview, commitAdapter, layer.id, t]
  );

  const rangeBeforeRef = useRef<[number, number] | null>(null);
  const handleRangeChange = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      if (value.length !== 2) {
        return;
      }
      if (
        !previewStructural({
          config: { adapter: { beginEndStepPct: [value[0]!, value[1]!] }, layerType: 'control' },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        })
      ) {
        return;
      }
      if (rangeBeforeRef.current === null) {
        rangeBeforeRef.current = adapter.beginEndStepPct;
      }
    },
    [adapter.beginEndStepPct, previewStructural, layer.id]
  );
  const handleRangeChangeEnd = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      const before = rangeBeforeRef.current ?? adapter.beginEndStepPct;
      rangeBeforeRef.current = null;
      if (value.length !== 2) {
        return;
      }
      commitAdapter(
        { beginEndStepPct: [value[0]!, value[1]!] },
        { beginEndStepPct: before },
        t('widgets.layers.control.stepRange')
      );
    },
    [adapter.beginEndStepPct, commitAdapter, t]
  );

  const handleTransparencyToggle = useCallback(
    ({ checked }: { checked: boolean }) => {
      commitPrepared(t('widgets.layers.control.transparencyEffect'), (model) =>
        model.prepare({
          before: { layerType: 'control', withTransparencyEffect: layer.withTransparencyEffect },
          config: { layerType: 'control', withTransparencyEffect: checked },
          id: layer.id,
          type: 'patch-config',
        })
      );
    },
    [commitPrepared, layer.id, layer.withTransparencyEffect, t]
  );

  const controlModeCollection = useMemo(
    () =>
      createListCollection({
        items: CONTROL_MODES.map((mode) => ({ label: t(`widgets.layers.control.modes.${mode}`), value: mode })),
      }),
    [t]
  );

  const kindValue = useMemo(() => [adapter.kind], [adapter.kind]);
  const modelValue = useMemo(() => (adapter.model ? [adapter.model] : []), [adapter.model]);
  const controlModeValue = useMemo(() => [adapter.controlMode ?? 'balanced'], [adapter.controlMode]);
  const rangeValue = useMemo(() => [...adapter.beginEndStepPct], [adapter.beginEndStepPct]);
  const rangeAria = useMemo(() => [t('widgets.layers.control.beginStep'), t('widgets.layers.control.endStep')], [t]);

  // The catalog name, not the filtered options: an unusable model still needs a name for the alert to refer to.
  const adapterModel = models.find((model) => model.key === adapter.model) ?? null;
  const hasContent = useCanvasEngineRead(engine, () => engine?.exports.hasExportableLayerContent(layer.id) ?? false);
  // Re-read the reason when the capability table changes; the engine read below consults it.
  const capabilitiesRevision = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    selectCapabilitiesRevision
  );
  // Judged as the invocation judges it: earlier contributing control layers with content claim limited slots first,
  // and only layers that pass claim one.
  const validationReason = useCanvasEngineRead(engine, () => {
    void capabilitiesRevision;
    if (!mainModel) {
      return null;
    }
    const leaves = engine?.document.model()?.compileLeaves() ?? [];
    const index = leaves.findIndex((leaf) => leaf.layer.id === layer.id);
    const contributing = engine ? (leaves[index]?.contributionEnabled ?? false) : layer.isEnabled;
    if (!contributing) {
      return null;
    }
    const earlier = leaves
      .slice(0, Math.max(0, index))
      .filter((leaf) => leaf.contributionEnabled && engine?.exports.hasExportableLayerContent(leaf.id))
      .map((leaf) => leaf.layer)
      .filter((candidate): candidate is CanvasControlLayerContract => candidate.type === 'control');
    return getControlLayerReasonInSequence({ earlier, layer, mainModel, models });
  });
  // Content-dependent problems only matter once the layer has pixels, but a
  // missing model deserves the warning even on a fresh empty layer. A missing
  // capability table is not a problem with this layer and is shown on its own.
  const capabilitiesUnavailable = validationReason === 'capabilities_unavailable';
  // The click flips the status to `loading` synchronously; keeping the failure surface mounted for the
  // retry it started keeps the button, and the user's focus, in place.
  const isRetryingCapabilities = hasRequestedCapabilitiesRetry && capabilitiesStatus === 'loading';
  const showCapabilitiesFailure = capabilitiesUnavailable && (capabilitiesStatus === 'error' || isRetryingCapabilities);
  // A missing model or a kind to switch is worth saying on a fresh empty layer; an empty model list explains itself.
  const visibleValidationReason =
    validationReason &&
    !capabilitiesUnavailable &&
    (hasContent || validationReason === 'missing_model' || validationReason === 'switch_adapter_kind') &&
    !(validationReason === 'missing_model' && hiddenModelReasons.length > 0)
      ? validationReason
      : null;

  return (
    <Stack ref={rootRef} gap="2">
      <HStack gap="2">
        <Field flex="1" label={t('widgets.layers.control.kind')} minW="0">
          <Select
            aria-label={t('widgets.layers.control.kind')}
            collection={kindCollection}
            positioning={SELECT_POSITIONING}
            value={kindValue}
            valueText={t(`widgets.layers.control.kinds.${adapter.kind}`)}
            onValueChange={handleKindChange}
          />
        </Field>
      </HStack>
      <Field
        helpText={
          hiddenModelReasons.length > 0
            ? hiddenModelReasons.map((reason) => t(`widgets.layers.control.hiddenModels.${reason}`)).join(' ')
            : undefined
        }
        label={t('widgets.layers.control.model')}
      >
        <Select
          aria-label={t('widgets.layers.control.model')}
          collection={modelCollection}
          positioning={SELECT_POSITIONING}
          value={modelValue}
          valueText={adapterModel?.name ?? t('widgets.layers.control.selectModel')}
          valueTextProps={adapter.model ? undefined : MISSING_MODEL_VALUE_TEXT_PROPS}
          onValueChange={handleModelChange}
        />
      </Field>
      <ScrubberField
        defaultValue={CONTROL_ADAPTER_DEFAULTS[adapter.kind].weight}
        formatValue={formatWeight}
        inputMax={CONTROL_WEIGHT_BOUNDS.inputMax}
        inputMin={weightInputMin}
        label={t('widgets.layers.control.weight')}
        max={CONTROL_WEIGHT_BOUNDS.sliderMax}
        min={CONTROL_WEIGHT_BOUNDS.sliderMin}
        step={CONTROL_WEIGHT_BOUNDS.step}
        value={adapter.weight}
        onChange={handleWeightChange}
        onChangeEnd={handleWeightChangeEnd}
      />
      <Field label={t('widgets.layers.control.stepRange')}>
        <Slider
          aria-label={rangeAria}
          formatValue={formatUnitPercent}
          max={1}
          min={0}
          step={0.01}
          value={rangeValue}
          withThumbTooltip
          onValueChange={handleRangeChange}
          onValueChangeEnd={handleRangeChangeEnd}
        />
      </Field>
      {adapter.kind === 'controlnet' ? (
        <Field label={t('widgets.layers.control.mode')}>
          <Select
            aria-label={t('widgets.layers.control.mode')}
            collection={controlModeCollection}
            positioning={SELECT_POSITIONING}
            value={controlModeValue}
            valueText={t(`widgets.layers.control.modes.${adapter.controlMode ?? 'balanced'}`)}
            onValueChange={handleControlModeChange}
          />
        </Field>
      ) : null}
      <Switch.Root
        checked={layer.withTransparencyEffect}
        colorPalette="accent"
        size="sm"
        onCheckedChange={handleTransparencyToggle}
      >
        <Switch.HiddenInput />
        <Switch.Control>
          <Switch.Thumb />
        </Switch.Control>
        <Switch.Label>
          <Text fontSize="md">{t('widgets.layers.control.transparencyEffect')}</Text>
        </Switch.Label>
      </Switch.Root>
      <LayerFilterOperationButton
        engine={engine}
        layer={layer}
        onOperationStarted={onOperationStarted}
        operations={engine ? getCanvasOperations(engine) : null}
      />
      {visibleValidationReason ? (
        <HStack align="center" gap="2" role="alert">
          <Text color="fg.warning" flex="1" fontSize="xs">
            {describeControlLayerReason(t, visibleValidationReason, suggestedKind)}
          </Text>
          {visibleValidationReason === 'switch_adapter_kind' && suggestedKind ? (
            <Button flexShrink={0} variant="outline" onClick={handleSwitchToSuggestedKind}>
              {t('widgets.layers.control.switchKind', {
                kind: t(`widgets.layers.control.kinds.${suggestedKind}`),
              })}
            </Button>
          ) : null}
        </HStack>
      ) : null}
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

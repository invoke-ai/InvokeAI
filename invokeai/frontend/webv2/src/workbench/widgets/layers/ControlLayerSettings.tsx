import type {
  NumberInput as ChakraNumberInput,
  SelectValueChangeDetails,
  SliderValueChangeDetails,
} from '@chakra-ui/react';
import type { ArchitectureCapabilitiesSnapshot } from '@features/generation/runtime';
import type {
  CanvasControlAdapterContract,
  CanvasControlLayerContract,
  CanvasDocumentCapability,
  CanvasLayerContract,
} from '@workbench/canvas-engine/api';
import type { LayerFilterOperationEngine } from '@workbench/widgets/layers/LayerFilterOperationButton';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';

import { createListCollection, HStack, NumberInput, Stack, Switch, Text } from '@chakra-ui/react';
import {
  getControlValidationReason,
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
import { lookupDocumentLeaf } from '@workbench/canvas-engine/api';
import { getCanvasOperations, resolveDefaultFilterForModel } from '@workbench/canvas-operations/api';
import { useCanvasEngineRead } from '@workbench/widgets/canvas/engineStoreHooks';
import { useStructuralPreview } from '@workbench/widgets/canvas/useStructuralCommit';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { getCompatibleControlModels } from './controlModelOptions';
import { LayerFilterOperationButton } from './LayerFilterOperationButton';
import { CONTROL_ADAPTER_DEFAULTS, CONTROL_WEIGHT_BOUNDS } from './layerOps';
import { runLayerFilterOperation } from './layerPropertiesOperation';
import { useSelectedMainModel } from './useSelectedMainModel';

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: true } as const;

const selectCapabilitiesStatus = (snapshot: ArchitectureCapabilitiesSnapshot) => snapshot.status;

const CONTROL_ADAPTER_KINDS: readonly ControlAdapterKind[] = [
  'controlnet',
  't2i_adapter',
  'control_lora',
  'z_image_control',
];
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
/** Contributing control leaves of one adapter kind with content, in generation order. */
const contributingControlLayers = (
  engine: CanvasStructuralEngine & LayerFilterOperationEngine & { readonly document: CanvasDocumentCapability },
  kind: CanvasControlAdapterContract['kind']
): CanvasLayerContract[] =>
  (engine.document.model()?.compileLeaves() ?? [])
    .filter(
      (leaf) =>
        leaf.contributionEnabled &&
        leaf.layer.type === 'control' &&
        leaf.layer.adapter.kind === kind &&
        engine.exports.hasExportableLayerContent(leaf.id)
    )
    .map((leaf) => leaf.layer);

export const ControlLayerSettings = ({ engine, layer, onOperationStarted }: ControlLayerSettingsProps) => {
  const { t } = useTranslation();
  const { commit: commitPrepared, preview: previewStructural } = useStructuralPreview(engine);
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
          base ? isControlKindSupportedForBase(base, kind) : kind !== 'z_image_control'
        ),
      [base]
    )
  );
  const kindCollection = useMemo(
    () =>
      createListCollection({
        items: kindOptions.map((kind) => ({ label: t(`widgets.layers.control.kinds.${kind}`), value: kind })),
      }),
    [kindOptions, t]
  );

  // Adapter models matching the current kind + base (mirrors the generate model list).
  const modelOptions = useMemo(
    () => getCompatibleControlModels(models, base, adapter.kind),
    [adapter.kind, base, models]
  );
  const modelCollection = useMemo(
    () => createListCollection({ items: modelOptions.map((model) => ({ label: model.name, value: model.key })) }),
    [modelOptions]
  );

  const handleKindChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const kind = value[0] as ControlAdapterKind | undefined;
      if (!kind || kind === adapter.kind) {
        return;
      }
      // Switching kind clears the model (its base/type no longer matches) and, for
      // non-ControlNet kinds, drops the control mode.
      const defaults = CONTROL_ADAPTER_DEFAULTS[kind];
      commitAdapter(
        { ...defaults, beginEndStepPct: [...defaults.beginEndStepPct] },
        { ...adapter, beginEndStepPct: [...adapter.beginEndStepPct] },
        t('widgets.layers.control.kind')
      );
    },
    [adapter, commitAdapter, t]
  );

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

  const weightBeforeRef = useRef<number | null>(null);
  const handleWeightChange = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      const next = value[0];
      if (next === undefined || !Number.isFinite(next)) {
        return;
      }
      if (
        !previewStructural({
          config: { adapter: { weight: next }, layerType: 'control' },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        })
      ) {
        return;
      }
      if (weightBeforeRef.current === null) {
        weightBeforeRef.current = adapter.weight;
      }
    },
    [adapter.weight, previewStructural, layer.id]
  );
  const handleWeightChangeEnd = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      const next = value[0];
      const before = weightBeforeRef.current ?? adapter.weight;
      weightBeforeRef.current = null;
      if (next === undefined || !Number.isFinite(next)) {
        return;
      }
      commitAdapter({ weight: next }, { weight: before }, t('widgets.layers.control.weight'));
    },
    [adapter.weight, commitAdapter, t]
  );
  const handleWeightInputChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (
        !Number.isFinite(valueAsNumber) ||
        valueAsNumber < weightInputMin ||
        valueAsNumber > CONTROL_WEIGHT_BOUNDS.inputMax
      ) {
        return;
      }
      commitAdapter({ weight: valueAsNumber }, { weight: adapter.weight }, t('widgets.layers.control.weight'));
    },
    [adapter.weight, commitAdapter, t, weightInputMin]
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
  const weightValue = useMemo(
    () => [Math.min(CONTROL_WEIGHT_BOUNDS.sliderMax, Math.max(CONTROL_WEIGHT_BOUNDS.sliderMin, adapter.weight))],
    [adapter.weight]
  );
  const weightInputValue = String(adapter.weight);
  const rangeValue = useMemo(() => [...adapter.beginEndStepPct], [adapter.beginEndStepPct]);
  const weightAria = useMemo(() => [t('widgets.layers.control.weight')], [t]);
  const rangeAria = useMemo(() => [t('widgets.layers.control.beginStep'), t('widgets.layers.control.endStep')], [t]);

  const selectedModelName = modelOptions.find((model) => model.key === adapter.model)?.name;
  const adapterModel = models.find((model) => model.key === adapter.model) ?? null;
  const hasContent = useCanvasEngineRead(engine, () => engine?.exports.hasExportableLayerContent(layer.id) ?? false);
  const controlLoraIndex = useCanvasEngineRead(engine, () =>
    adapter.kind === 'control_lora' && engine
      ? contributingControlLayers(engine, 'control_lora').findIndex((candidate) => candidate.id === layer.id)
      : 0
  );
  const zImageControlIndex = useCanvasEngineRead(engine, () =>
    adapter.kind === 'z_image_control' && engine
      ? contributingControlLayers(engine, 'z_image_control').findIndex((candidate) => candidate.id === layer.id)
      : 0
  );
  const contributing = useCanvasEngineRead(engine, () =>
    engine
      ? (lookupDocumentLeaf(engine.document.model()?.document, layer.id)?.contributionEnabled ?? false)
      : layer.isEnabled
  );
  // Same reason as `kindOptions`: validation asks the capability table whether the kind is supported.
  const validationReason = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(
      () =>
        contributing && mainModel
          ? getControlValidationReason({
              adapterModel: adapterModel ? { base: adapterModel.base, type: adapterModel.type } : null,
              beginEndStepPct: adapter.beginEndStepPct,
              controlLoraIndex: Math.max(0, controlLoraIndex),
              kind: adapter.kind,
              mainBase: mainModel.base,
              mainVariant: mainModel.variant ?? undefined,
              weight: adapter.weight,
              zImageControlIndex: Math.max(0, zImageControlIndex),
            })
          : null,
      [
        adapter.beginEndStepPct,
        adapter.kind,
        adapter.weight,
        adapterModel,
        contributing,
        controlLoraIndex,
        mainModel,
        zImageControlIndex,
      ]
    )
  );
  // Content-dependent problems only matter once the layer has pixels, but a
  // missing model deserves the warning even on a fresh empty layer. A missing
  // capability table is not a problem with this layer and is shown on its own.
  const capabilitiesUnavailable = validationReason === 'capabilities_unavailable';
  // The click flips the status to `loading` synchronously; keeping the failure surface mounted for the
  // retry it started keeps the button, and the user's focus, in place.
  const isRetryingCapabilities = hasRequestedCapabilitiesRetry && capabilitiesStatus === 'loading';
  const showCapabilitiesFailure = capabilitiesUnavailable && (capabilitiesStatus === 'error' || isRetryingCapabilities);
  const visibleValidationReason =
    validationReason && !capabilitiesUnavailable && (hasContent || validationReason === 'missing_model')
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
            size="xs"
            value={kindValue}
            valueText={t(`widgets.layers.control.kinds.${adapter.kind}`)}
            onValueChange={handleKindChange}
          />
        </Field>
      </HStack>
      <Field label={t('widgets.layers.control.model')}>
        <Select
          aria-label={t('widgets.layers.control.model')}
          collection={modelCollection}
          positioning={SELECT_POSITIONING}
          size="xs"
          value={modelValue}
          valueText={selectedModelName ?? t('widgets.layers.control.selectModel')}
          valueTextProps={adapter.model ? undefined : MISSING_MODEL_VALUE_TEXT_PROPS}
          onValueChange={handleModelChange}
        />
      </Field>
      <Field label={t('widgets.layers.control.weight')}>
        <HStack gap="2">
          <Slider
            aria-label={weightAria}
            flex="1"
            formatValue={formatWeight}
            max={CONTROL_WEIGHT_BOUNDS.sliderMax}
            min={CONTROL_WEIGHT_BOUNDS.sliderMin}
            size="sm"
            step={CONTROL_WEIGHT_BOUNDS.step}
            value={weightValue}
            withThumbTooltip
            onValueChange={handleWeightChange}
            onValueChangeEnd={handleWeightChangeEnd}
          />
          <NumberInput.Root
            max={CONTROL_WEIGHT_BOUNDS.inputMax}
            min={weightInputMin}
            size="xs"
            step={CONTROL_WEIGHT_BOUNDS.step}
            value={weightInputValue}
            w="20"
            onValueChange={handleWeightInputChange}
          >
            <NumberInput.Control />
            <NumberInput.Input aria-label={t('widgets.layers.control.weight')} />
          </NumberInput.Root>
        </HStack>
      </Field>
      <Field label={t('widgets.layers.control.stepRange')}>
        <Slider
          aria-label={rangeAria}
          formatValue={formatUnitPercent}
          max={1}
          min={0}
          size="sm"
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
            size="xs"
            value={controlModeValue}
            valueText={t(`widgets.layers.control.modes.${adapter.controlMode ?? 'balanced'}`)}
            onValueChange={handleControlModeChange}
          />
        </Field>
      ) : null}
      <Switch.Root
        checked={layer.withTransparencyEffect}
        colorPalette="accent"
        size="xs"
        onCheckedChange={handleTransparencyToggle}
      >
        <Switch.HiddenInput />
        <Switch.Control>
          <Switch.Thumb />
        </Switch.Control>
        <Switch.Label>
          <Text fontSize="xs">{t('widgets.layers.control.transparencyEffect')}</Text>
        </Switch.Label>
      </Switch.Root>
      <LayerFilterOperationButton
        engine={engine}
        layer={layer}
        onOperationStarted={onOperationStarted}
        operations={engine ? getCanvasOperations(engine) : null}
      />
      {visibleValidationReason ? (
        <Text color="fg.warning" fontSize="2xs" role="alert">
          {t(`widgets.layers.control.validation.${visibleValidationReason}`)}
        </Text>
      ) : null}
      {showCapabilitiesFailure ? (
        <HStack ref={handOverFocusOnLoad} aria-busy={isRetryingCapabilities} gap="2" role="alert">
          <Text color="fg.warning" flex="1" fontSize="2xs">
            {t('widgets.layers.control.capabilitiesLoadFailed')}
          </Text>
          {/* `aria-disabled` rather than `disabled`: a disabled button drops the focus it holds. */}
          <Button
            aria-busy={isRetryingCapabilities}
            aria-disabled={isRetryingCapabilities}
            size="xs"
            variant="outline"
            onClick={retryCapabilities}
          >
            {t('common.retry')}
          </Button>
        </HStack>
      ) : null}
      {capabilitiesUnavailable && !showCapabilitiesFailure ? (
        <Text color="fg.muted" fontSize="2xs" role="status">
          {t('widgets.layers.control.capabilitiesLoading')}
        </Text>
      ) : null}
    </Stack>
  );
};

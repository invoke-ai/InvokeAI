import type {
  NumberInput as ChakraNumberInput,
  SelectValueChangeDetails,
  SliderValueChangeDetails,
} from '@chakra-ui/react';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { ControlAdapterKind } from '@workbench/generation/canvas/addControlLayers';
import type { CanvasControlAdapterContract, CanvasControlLayerContract } from '@workbench/types';

import { createListCollection, HStack, NumberInput, Stack, Switch, Text } from '@chakra-ui/react';
import { Button, Field, Select, Slider } from '@workbench/components/ui';
import { isControlKindSupportedForBase } from '@workbench/generation/canvas/addControlLayers';
import { resolveDefaultFilterForModel } from '@workbench/generation/canvas/controlRecommendations';
import { getControlValidationReason } from '@workbench/generation/canvas/controlValidation';
import { useModelsSelector } from '@workbench/models/modelsStore';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { getCompatibleControlModels } from './controlModelOptions';
import { applyStructural, applyStructuralPreview, CONTROL_ADAPTER_DEFAULTS, CONTROL_WEIGHT_BOUNDS } from './layerOps';
import { runLayerFilterOperation } from './layerPropertiesOperation';

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: false } as const;

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

/** The main model's base, read from the generate widget values (drives adapter support). */
const useSelectedMainModel = () => {
  const modelKey = useActiveProjectSelector((project) => {
    const values = getProjectWidgetValues(project, 'generate');
    const model = values?.model;
    return model && typeof model === 'object' && 'key' in model ? String((model as { key: unknown }).key) : null;
  });
  const models = useModelsSelector((snapshot) => snapshot.models);
  return useMemo(() => models.find((model) => model.key === modelKey) ?? null, [models, modelKey]);
};

interface ControlLayerSettingsProps {
  engine: CanvasEngine | null;
  layer: CanvasControlLayerContract;
  onOperationStarted(): void;
}

/**
 * Per-layer settings for a selected control layer (plan §1.3): adapter kind +
 * model, weight, begin/end step range, control mode (ControlNet only), the
 * transparency effect toggle, and a non-destructive filter section (type +
 * per-type settings + preview/apply/cancel). Adapter edits go through the canvas
 * undo stack (`updateCanvasLayerConfig`); the filter preview runs on the utility
 * queue and never mutates the document until "Apply".
 */
export const ControlLayerSettings = ({ engine, layer, onOperationStarted }: ControlLayerSettingsProps) => {
  const { t } = useTranslation();
  const dispatch = useWorkbenchDispatch();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const mainModel = useSelectedMainModel();
  const base = mainModel?.base ?? null;
  const { adapter } = layer;
  const weightInputMin = adapter.kind === 'z_image_control' ? 0 : CONTROL_WEIGHT_BOUNDS.inputMin;

  const commitAdapter = useCallback(
    (next: Partial<CanvasControlAdapterContract>, before: Partial<CanvasControlAdapterContract>, label: string) => {
      applyStructural(
        engine,
        dispatch,
        label,
        { config: { adapter: next, layerType: 'control' }, id: layer.id, type: 'updateCanvasLayerConfig' },
        { config: { adapter: before, layerType: 'control' }, id: layer.id, type: 'updateCanvasLayerConfig' }
      );
    },
    [dispatch, engine, layer.id]
  );

  // Adapter kinds supported by the selected base. Z-Image Control is only shown
  // when a compatible Z-Image main model is selected.
  const kindOptions = useMemo(
    () =>
      CONTROL_ADAPTER_KINDS.filter((kind) =>
        base ? isControlKindSupportedForBase(base, kind) : kind !== 'z_image_control'
      ),
    [base]
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
          runLayerFilterOperation(() => engine?.startFilterOperation(layer.id, recommendation), onOperationStarted);
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
        !applyStructuralPreview(engine, dispatch, {
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
    [adapter.weight, dispatch, engine, layer.id]
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
        !applyStructuralPreview(engine, dispatch, {
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
    [adapter.beginEndStepPct, dispatch, engine, layer.id]
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
      applyStructural(
        engine,
        dispatch,
        t('widgets.layers.control.transparencyEffect'),
        {
          config: { layerType: 'control', withTransparencyEffect: checked },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        },
        {
          config: { layerType: 'control', withTransparencyEffect: !checked },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        }
      );
    },
    [dispatch, engine, layer.id, t]
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
  const hasContent = engine?.hasExportableLayerContent(layer.id) ?? false;
  const controlLoraIndex =
    adapter.kind === 'control_lora' && engine
      ? (engine
          .getDocument()
          ?.layers.filter(
            (candidate) =>
              candidate.isEnabled &&
              candidate.type === 'control' &&
              candidate.adapter.kind === 'control_lora' &&
              engine.hasExportableLayerContent(candidate.id)
          )
          .findIndex((candidate) => candidate.id === layer.id) ?? 0)
      : 0;
  const zImageControlIndex =
    adapter.kind === 'z_image_control' && engine
      ? (engine
          .getDocument()
          ?.layers.filter(
            (candidate) =>
              candidate.isEnabled &&
              candidate.type === 'control' &&
              candidate.adapter.kind === 'z_image_control' &&
              engine.hasExportableLayerContent(candidate.id)
          )
          .findIndex((candidate) => candidate.id === layer.id) ?? 0)
      : 0;
  const validationReason =
    layer.isEnabled && hasContent && mainModel
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
      : null;
  const startFilter = useCallback(
    () => runLayerFilterOperation(() => engine?.startFilterOperation(layer.id), onOperationStarted),
    [engine, layer.id, onOperationStarted]
  );

  return (
    <Stack borderColor="border.subtle" borderWidth="1px" gap="2" p="2" rounded="md">
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
      <Button disabled={!engine || layer.isLocked} size="xs" variant="outline" onClick={startFilter}>
        {t('widgets.layers.control.filter')}
      </Button>
      {validationReason ? (
        <Text color="fg.warning" fontSize="2xs" role="alert">
          {t(`widgets.layers.control.validation.${validationReason}`)}
        </Text>
      ) : null}
    </Stack>
  );
};

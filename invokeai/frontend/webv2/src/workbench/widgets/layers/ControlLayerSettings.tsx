import type { SelectValueChangeDetails, SliderValueChangeDetails } from '@chakra-ui/react';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { ControlAdapterKind } from '@workbench/generation/canvas/addControlLayers';
import type { FilterParamSpec } from '@workbench/generation/canvas/filterGraphs';
import type { CanvasControlAdapterContract, CanvasControlLayerContract } from '@workbench/types';

import { createListCollection, HStack, Stack, Switch, Text } from '@chakra-ui/react';
import { socketHub } from '@workbench/backend/socketHub';
import { runUtilityGraph } from '@workbench/canvas-engine/backend/utilityQueue';
import { applyToPoint, fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { Button, Field, Select, Slider } from '@workbench/components/ui';
import { makeImageDurable } from '@workbench/gallery/api';
import { isControlKindSupportedForBase } from '@workbench/generation/canvas/addControlLayers';
import {
  buildFilterDefaults,
  CONTROL_FILTERS,
  DEFAULT_CONTROL_FILTER_TYPE,
  getFilterDefinition,
} from '@workbench/generation/canvas/filterGraphs';
import { useModelsSelector } from '@workbench/models/modelsStore';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { runControlFilterPreview } from './controlFilterPreview';
import { applyStructural } from './layerOps';

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: false } as const;

const CONTROL_ADAPTER_KINDS: readonly ControlAdapterKind[] = ['controlnet', 't2i_adapter', 'control_lora'];
const CONTROL_MODES: readonly NonNullable<CanvasControlAdapterContract['controlMode']>[] = [
  'balanced',
  'more_prompt',
  'more_control',
  'unbalanced',
];

const formatUnitPercent = (value: number): string => `${Math.round(value * 100)}%`;
const formatWeight = (value: number): string => value.toFixed(2);

/** The main model's base, read from the generate widget values (drives adapter support). */
const useSelectedModelBase = (): string | null => {
  const modelKey = useActiveProjectSelector((project) => {
    const values = getProjectWidgetValues(project, 'generate');
    const model = values?.model;
    return model && typeof model === 'object' && 'key' in model ? String((model as { key: unknown }).key) : null;
  });
  const models = useModelsSelector((snapshot) => snapshot.models);
  return useMemo(() => models.find((model) => model.key === modelKey)?.base ?? null, [models, modelKey]);
};

interface ControlLayerSettingsProps {
  engine: CanvasEngine | null;
  filterRequestToken?: number | null;
  layer: CanvasControlLayerContract;
}

/**
 * Per-layer settings for a selected control layer (plan §1.3): adapter kind +
 * model, weight, begin/end step range, control mode (ControlNet only), the
 * transparency effect toggle, and a non-destructive filter section (type +
 * per-type settings + preview/apply/cancel). Adapter edits go through the canvas
 * undo stack (`updateCanvasLayerConfig`); the filter preview runs on the utility
 * queue and never mutates the document until "Apply".
 */
export const ControlLayerSettings = ({ engine, filterRequestToken = null, layer }: ControlLayerSettingsProps) => {
  const { t } = useTranslation();
  const dispatch = useWorkbenchDispatch();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const base = useSelectedModelBase();
  const { adapter } = layer;

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

  // Adapter kinds supported by the selected base (legacy support matrix). With no
  // model selected, offer all kinds so the layer is still configurable.
  const kindOptions = useMemo(
    () => CONTROL_ADAPTER_KINDS.filter((kind) => !base || isControlKindSupportedForBase(base, kind)),
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
    () => models.filter((model) => model.type === adapter.kind && (!base || model.base === base)),
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
      commitAdapter(
        { controlMode: kind === 'controlnet' ? (adapter.controlMode ?? 'balanced') : null, kind, model: null },
        { controlMode: adapter.controlMode, kind: adapter.kind, model: adapter.model },
        t('widgets.layers.control.kind')
      );
    },
    [adapter.controlMode, adapter.kind, adapter.model, commitAdapter, t]
  );

  const handleModelChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const model = value[0] ?? null;
      if (model !== adapter.model) {
        commitAdapter({ model }, { model: adapter.model }, t('widgets.layers.control.model'));
      }
    },
    [adapter.model, commitAdapter, t]
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
      if (weightBeforeRef.current === null) {
        weightBeforeRef.current = adapter.weight;
      }
      dispatch({
        config: { adapter: { weight: next }, layerType: 'control' },
        id: layer.id,
        type: 'updateCanvasLayerConfig',
      });
    },
    [adapter.weight, dispatch, layer.id]
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

  const rangeBeforeRef = useRef<[number, number] | null>(null);
  const handleRangeChange = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      if (value.length !== 2) {
        return;
      }
      if (rangeBeforeRef.current === null) {
        rangeBeforeRef.current = adapter.beginEndStepPct;
      }
      dispatch({
        config: { adapter: { beginEndStepPct: [value[0]!, value[1]!] }, layerType: 'control' },
        id: layer.id,
        type: 'updateCanvasLayerConfig',
      });
    },
    [adapter.beginEndStepPct, dispatch, layer.id]
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
  const weightValue = useMemo(() => [adapter.weight], [adapter.weight]);
  const rangeValue = useMemo(() => [...adapter.beginEndStepPct], [adapter.beginEndStepPct]);
  const weightAria = useMemo(() => [t('widgets.layers.control.weight')], [t]);
  const rangeAria = useMemo(() => [t('widgets.layers.control.beginStep'), t('widgets.layers.control.endStep')], [t]);

  const selectedModelName = modelOptions.find((model) => model.key === adapter.model)?.name;

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
        <Slider
          aria-label={weightAria}
          formatValue={formatWeight}
          max={2}
          min={-1}
          size="sm"
          step={0.01}
          value={weightValue}
          withThumbTooltip
          onValueChange={handleWeightChange}
          onValueChangeEnd={handleWeightChangeEnd}
        />
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
      {/*
       * Keyed on layer id (Rule 5, no-use-effect): switching the selected
       * control layer swaps `layer` without unmounting `ControlLayerSettings`
       * itself, so without a key the filter section would keep its local
       * `preview`/`abortRef` state and the outgoing layer's on-canvas preview
       * would never get torn down. Keying forces a fresh instance per layer,
       * which both resets local state and runs the outgoing instance's
       * unmount cleanup (see its ref-callback below).
       */}
      <ControlFilterSection
        key={`${layer.id}-${filterRequestToken ?? 'default'}`}
        dispatch={dispatch}
        engine={engine}
        focusFilter={filterRequestToken !== null}
        layer={layer}
      />
    </Stack>
  );
};

// ---- Filter section (non-destructive preview / apply / cancel) --------------

type FilterPreviewState =
  | { status: 'idle' }
  | { status: 'running' }
  | { status: 'previewing'; imageName: string; width: number; height: number; origin: { x: number; y: number } }
  | { status: 'error'; message: string };

interface ControlFilterSectionProps {
  dispatch: ReturnType<typeof useWorkbenchDispatch>;
  engine: CanvasEngine | null;
  focusFilter: boolean;
  layer: CanvasControlLayerContract;
}

/** True when a control layer has rasterizable content to run a filter over. */
const hasFilterableContent = (layer: CanvasControlLayerContract): boolean =>
  layer.source.type === 'image' || (layer.source.type === 'paint' && layer.source.bitmap !== null);

const ControlFilterSection = ({ dispatch, engine, focusFilter, layer }: ControlFilterSectionProps) => {
  const { t } = useTranslation();
  const [preview, setPreview] = useState<FilterPreviewState>({ status: 'idle' });
  const abortRef = useRef<AbortController | null>(null);

  const filterType = layer.filter?.type ?? DEFAULT_CONTROL_FILTER_TYPE;
  const definition = getFilterDefinition(filterType);
  const settings = useMemo(
    () => layer.filter?.settings ?? (definition ? buildFilterDefaults(definition) : {}),
    [definition, layer.filter?.settings]
  );

  const commitFilter = useCallback(
    (type: string, nextSettings: Record<string, unknown>) => {
      applyStructural(
        engine,
        dispatch,
        t('widgets.layers.control.filter'),
        {
          config: { filter: { settings: nextSettings, type }, layerType: 'control' },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        },
        { config: { filter: layer.filter, layerType: 'control' }, id: layer.id, type: 'updateCanvasLayerConfig' }
      );
    },
    [dispatch, engine, layer.filter, layer.id, t]
  );

  const filterCollection = useMemo(
    () =>
      createListCollection({
        items: CONTROL_FILTERS.map((filter) => ({
          label: t(`widgets.layers.control.filters.${filter.type}`, filter.type),
          value: filter.type,
        })),
      }),
    [t]
  );

  const clearPreview = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    engine?.setFilterPreview(layer.id, null);
    setPreview({ status: 'idle' });
  }, [engine, layer.id]);

  /**
   * React 19 ref-callback cleanup (the established convention here — see
   * `CanvasSurface`'s `bindContainer` — rather than a `useEffect`): the
   * returned function fires when this DOM node unmounts, which happens
   * whenever this section unmounts (layer deselected) or — since the parent
   * keys this component by `layer.id` — whenever the selected control layer
   * changes to a different one. It aborts any in-flight filter run and
   * dismisses this layer's on-canvas preview, so switching away or deleting
   * the layer never strands a preview with no way to dismiss it.
   */
  const sectionRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (!node) {
        return;
      }
      return () => {
        abortRef.current?.abort();
        abortRef.current = null;
        engine?.setFilterPreview(layer.id, null);
      };
    },
    [engine, layer.id]
  );

  const handleFilterTypeChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const type = value[0];
      if (!type || type === filterType) {
        return;
      }
      clearPreview();
      const def = getFilterDefinition(type);
      commitFilter(type, def ? buildFilterDefaults(def) : {});
    },
    [clearPreview, commitFilter, filterType]
  );

  const handleSettingChange = useCallback(
    (key: string, value: unknown) => {
      commitFilter(filterType, { ...settings, [key]: value });
    },
    [commitFilter, filterType, settings]
  );

  const handlePreview = useCallback(async () => {
    if (!engine) {
      return;
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setPreview({ status: 'running' });
    try {
      const result = await runControlFilterPreview({
        deps: {
          executorDeps: { ...engine.getCompositeExecutorDeps(), dedupe: { byHash: new Map(), byKey: new Map() } },
          flushPendingUploads: () => engine.flushPendingUploads(),
          getDocument: () => engine.getDocument(),
          runFilterGraph: (graph, outputNodeId, signal) =>
            runUtilityGraph({ graph, hub: socketHub, outputNodeId, signal }).then((r) => r.imageName),
        },
        filterType,
        layerId: layer.id,
        settings,
        signal: controller.signal,
      });
      if (controller.signal.aborted) {
        return;
      }
      engine.setFilterPreview(layer.id, { imageName: result.imageName });
      setPreview({
        height: result.height,
        imageName: result.imageName,
        origin: result.origin,
        status: 'previewing',
        width: result.width,
      });
    } catch (error) {
      if (controller.signal.aborted) {
        return;
      }
      setPreview({ message: error instanceof Error ? error.message : String(error), status: 'error' });
    }
  }, [engine, filterType, layer.id, settings]);

  const handleApply = useCallback(async () => {
    if (preview.status !== 'previewing') {
      return;
    }
    const image = { height: preview.height, imageName: preview.imageName, width: preview.width };
    // The preview image is intermediate (so previews don't litter the gallery) —
    // promote the chosen one to durable BEFORE the swap, so the layer never ends
    // up pointing at an image the backend garbage-collects. A failed PATCH leaves
    // the document untouched and surfaces an error.
    try {
      await makeImageDurable(image.imageName);
    } catch (error) {
      setPreview({ message: error instanceof Error ? error.message : String(error), status: 'error' });
      return;
    }
    // Swap the layer source to the filtered image (undoable); clear the preview.
    applyStructural(
      engine,
      dispatch,
      t('widgets.layers.control.applyFilter'),
      { id: layer.id, source: { image, type: 'image' }, type: 'updateCanvasLayerSource' },
      { id: layer.id, source: layer.source, type: 'updateCanvasLayerSource' }
    );
    // The filtered image occupies the layer's content rect, which may sit off the
    // layer-local origin (a content-sized paint control layer at a non-zero
    // offset). An `image` source renders at layer-local {0,0}, so fold the content
    // origin — mapped through the layer's current transform — into the transform
    // translation, keeping the applied pixels exactly where the preview drew them.
    // The common case (image-source control layer, origin {0,0}) is a no-op.
    if (preview.origin.x !== 0 || preview.origin.y !== 0) {
      const matrix = fromTRS(
        { x: layer.transform.x, y: layer.transform.y },
        layer.transform.rotation,
        layer.transform.scaleX,
        layer.transform.scaleY
      );
      const docOrigin = applyToPoint(matrix, preview.origin);
      applyStructural(
        engine,
        dispatch,
        t('widgets.layers.control.applyFilter'),
        { id: layer.id, patch: { transform: { x: docOrigin.x, y: docOrigin.y } }, type: 'updateCanvasLayer' },
        {
          id: layer.id,
          patch: { transform: { x: layer.transform.x, y: layer.transform.y } },
          type: 'updateCanvasLayer',
        }
      );
    }
    engine?.setFilterPreview(layer.id, null);
    abortRef.current = null;
    setPreview({ status: 'idle' });
  }, [dispatch, engine, layer.id, layer.source, layer.transform, preview, t]);

  const canPreview = !!engine && hasFilterableContent(layer) && preview.status !== 'running';
  const filterValue = useMemo(() => [filterType], [filterType]);
  const filterTriggerProps = useMemo(() => ({ autoFocus: focusFilter }), [focusFilter]);

  return (
    <Stack borderColor="border.subtle" borderTopWidth="1px" gap="2" pt="2" ref={sectionRef}>
      <Field label={t('widgets.layers.control.filter')}>
        <Select
          aria-label={t('widgets.layers.control.filter')}
          collection={filterCollection}
          positioning={SELECT_POSITIONING}
          size="xs"
          triggerProps={filterTriggerProps}
          value={filterValue}
          valueText={t(`widgets.layers.control.filters.${filterType}`, filterType)}
          onValueChange={handleFilterTypeChange}
        />
      </Field>
      {definition?.params.map((param) => (
        <FilterParamField key={param.key} param={param} value={settings[param.key]} onChange={handleSettingChange} />
      ))}
      <HStack gap="2">
        <Button
          disabled={!canPreview}
          flex="1"
          loading={preview.status === 'running'}
          size="xs"
          variant="outline"
          onClick={handlePreview}
        >
          {t('widgets.layers.control.preview')}
        </Button>
        {preview.status === 'previewing' ? (
          <>
            <Button colorPalette="accent" flex="1" size="xs" onClick={handleApply}>
              {t('widgets.layers.control.apply')}
            </Button>
            <Button flex="1" size="xs" variant="ghost" onClick={clearPreview}>
              {t('widgets.layers.control.cancel')}
            </Button>
          </>
        ) : null}
      </HStack>
      {preview.status === 'error' ? (
        <Text color="fg.error" fontSize="2xs">
          {preview.message}
        </Text>
      ) : null}
    </Stack>
  );
};

interface FilterParamFieldProps {
  param: FilterParamSpec;
  value: unknown;
  onChange: (key: string, value: unknown) => void;
}

/** One filter parameter editor (number slider / boolean switch / enum select). */
const FilterParamField = ({ param, value, onChange }: FilterParamFieldProps) => {
  const { t } = useTranslation();
  const label = t(`widgets.layers.control.filterParams.${param.key}`, param.key);
  const labelAria = useMemo(() => [label], [label]);

  const handleBoolean = useCallback(
    ({ checked }: { checked: boolean }) => onChange(param.key, checked),
    [onChange, param.key]
  );
  const handleEnum = useCallback(
    ({ value: next }: SelectValueChangeDetails) => {
      if (next[0]) {
        onChange(param.key, next[0]);
      }
    },
    [onChange, param.key]
  );
  const handleNumberEnd = useCallback(
    ({ value: next }: SliderValueChangeDetails) => {
      const n = next[0];
      if (n !== undefined && Number.isFinite(n)) {
        onChange(param.key, param.kind === 'number' && param.integer ? Math.round(n) : n);
      }
    },
    [onChange, param]
  );

  const enumCollection = useMemo(
    () =>
      param.kind === 'enum'
        ? createListCollection({ items: param.options.map((option) => ({ label: option, value: option })) })
        : null,
    [param]
  );
  const enumCurrent =
    param.kind === 'enum' && typeof value === 'string' && param.options.includes(value) ? value : param.default;
  const enumValue = useMemo(() => [String(enumCurrent)], [enumCurrent]);
  const numberCurrent = typeof value === 'number' && Number.isFinite(value) ? value : param.default;
  const numberValue = useMemo(() => [Number(numberCurrent)], [numberCurrent]);

  if (param.kind === 'boolean') {
    return (
      <Switch.Root
        checked={typeof value === 'boolean' ? value : param.default}
        colorPalette="accent"
        size="xs"
        onCheckedChange={handleBoolean}
      >
        <Switch.HiddenInput />
        <Switch.Control>
          <Switch.Thumb />
        </Switch.Control>
        <Switch.Label>
          <Text fontSize="xs">{label}</Text>
        </Switch.Label>
      </Switch.Root>
    );
  }

  if (param.kind === 'enum' && enumCollection) {
    return (
      <Field label={label}>
        <Select
          aria-label={label}
          collection={enumCollection}
          positioning={SELECT_POSITIONING}
          size="xs"
          value={enumValue}
          valueText={String(enumCurrent)}
          onValueChange={handleEnum}
        />
      </Field>
    );
  }

  return (
    <Field label={label}>
      <Slider
        aria-label={labelAria}
        max={param.kind === 'number' ? (param.max ?? 255) : 255}
        min={param.kind === 'number' ? (param.min ?? 0) : 0}
        size="sm"
        step={param.kind === 'number' ? (param.step ?? (param.integer ? 1 : 0.01)) : 0.01}
        value={numberValue}
        withThumbTooltip
        onValueChangeEnd={handleNumberEnd}
      />
    </Field>
  );
};

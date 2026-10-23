import type { GenerateLora, MainModelConfig } from '@features/generation/contracts';
import type { ProjectPromptDraftPatch } from '@features/generation/settings';
import type { ModelConfig, ModelTaxonomyType } from '@features/models';
import type { UpscaleComponentField, UpscaleComponentSlot } from '@features/upscale/core/settings';
import type { UpscaleMode, UpscaleWidgetValues } from '@features/upscale/core/types';
import type { FeatureHintId } from '@platform/ui/hints/hintRegistry';

import {
  Badge,
  Box,
  createListCollection,
  DataList,
  SegmentGroup,
  Separator,
  SimpleGrid,
  Stack,
  Text,
} from '@chakra-ui/react';
import {
  CapabilitiesGate,
  ConceptList,
  type ConceptModelPort,
  ConceptRow,
  GenerationSettingsSection,
  SeedField,
} from '@features/generation/components';
import {
  type ArchitectureCapabilitiesSnapshot,
  ensureArchitectureCapabilitiesLoaded,
  getArchitectureCapabilitiesSnapshot,
  subscribeArchitectureCapabilities,
} from '@features/generation/runtime';
import {
  getDefaultLoraWeight,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  isMainModelConfig,
  isModelIdentifierConfig,
  isVaeCompatibleWithGenerateModel,
  isVaeModelConfig,
} from '@features/generation/settings';
import {
  ensureModelsLoaded,
  getModelBaseColorPalette,
  getModelBaseLabel,
  getModelImageUrl,
  useModelsSelector,
  useOpenModelInManager,
} from '@features/models';
import { ModelSelect } from '@features/models/react';
import {
  coerceUpscaleScheduler,
  createDefaultUpscaleWidgetValues,
  getModelOnlyUpscalePlan,
  getSpandrelNativeScale,
  getUpscaleOutputDimensions,
  isSpandrelModelConfig,
  isSupportedUpscaleMainModel,
  isTileControlNetCandidate,
  normalizeUpscaleWidgetValues,
  selectUpscaleMainModel,
  upscaleArchitectureFor,
  syncUpscaleWidgetValuesWithModels,
  UPSCALE_CREATIVITY_MAX,
  UPSCALE_CREATIVITY_MIN,
  UPSCALE_PRESETS,
  UPSCALE_SCALE_MAX,
  UPSCALE_SCALE_MIN,
  UPSCALE_STRUCTURE_MAX,
  UPSCALE_STRUCTURE_MIN,
  UPSCALE_TILE_OVERLAP_MAX,
  UPSCALE_TILE_OVERLAP_MIN,
  UPSCALE_TILE_SIZE_MAX,
  UPSCALE_TILE_SIZE_MIN,
  upscaleComponentsFor,
  upscaleMaxCfgScale,
  upscaleMinSteps,
  upscaleSchedulerOptions,
} from '@features/upscale/core/settings';
import { SEED_MAX } from '@platform/core/seed';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Combobox, Field, Select, Tooltip } from '@platform/ui';
import { Button } from '@platform/ui/Button';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { SegmentedControl } from '@platform/ui/SegmentedControl';
import { toaster } from '@platform/ui/toaster';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { areInputImagesEquivalent, valuesAreEqual } from './upscaleComparators';
import { UpscalePromptFields } from './UpscaleFormFields';
import { UpscaleImageField } from './UpscaleImageField';
import { useUpscaleUi, useUpscaleUiActions } from './UpscaleUiContext';

/** Keep props stable across project patches so memoized form sections do not rerender for unrelated field edits. */

const VAE_PRECISION_COLLECTION = createListCollection({
  items: [
    { label: 'FP16', value: 'fp16' },
    { label: 'FP32', value: 'fp32' },
  ] as const,
});
const LARGE_OUTPUT_MEGAPIXELS = 50;
const DIMENSION_FORMATTER = new Intl.NumberFormat();
const MEGAPIXEL_FORMATTER = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 });

const SPANDREL_MODEL_TYPES: readonly ModelTaxonomyType[] = ['spandrel_image_to_image'];
const MAIN_MODEL_TYPES: readonly ModelTaxonomyType[] = ['main'];
const LORA_MODEL_TYPES: readonly ModelTaxonomyType[] = ['lora'];
const CONTROLNET_MODEL_TYPES: readonly ModelTaxonomyType[] = ['controlnet'];
const VAE_MODEL_TYPES: readonly ModelTaxonomyType[] = ['vae'];

/** How each encoder slot reads in the form; which slots show, and what fits them, is the core's call. */
const COMPONENT_FIELD_UI: Record<
  UpscaleComponentField,
  {
    hint: FeatureHintId;
    label: string;
    modelTypes: readonly ModelTaxonomyType[];
    placeholder: string;
    required: string;
  }
> = {
  clipEmbedModel: {
    hint: 'clipEmbed',
    label: 'widgets.upscale.clipEmbed',
    modelTypes: ['clip_embed'],
    placeholder: 'widgets.upscale.selectClipEmbed',
    required: 'widgets.upscale.clipEmbedRequired',
  },
  mistralEncoderModel: {
    hint: 'mistralEncoder',
    label: 'widgets.upscale.mistralEncoder',
    modelTypes: ['mistral_encoder'],
    placeholder: 'widgets.upscale.selectMistralEncoder',
    required: 'widgets.upscale.mistralEncoderRequired',
  },
  qwen35EncoderModel: {
    hint: 'qwen35Encoder',
    label: 'widgets.upscale.qwen35Encoder',
    modelTypes: ['qwen3_5_encoder'],
    placeholder: 'widgets.upscale.selectQwen35Encoder',
    required: 'widgets.upscale.qwen35EncoderRequired',
  },
  qwen3EncoderModel: {
    hint: 'qwen3Encoder',
    label: 'widgets.upscale.qwen3Encoder',
    modelTypes: ['qwen3_encoder'],
    placeholder: 'widgets.upscale.selectQwen3Encoder',
    required: 'widgets.upscale.qwen3EncoderRequired',
  },
  qwen3VlEncoderModel: {
    hint: 'qwen3VlEncoder',
    label: 'widgets.upscale.qwen3VlEncoder',
    modelTypes: ['qwen3_vl_encoder'],
    placeholder: 'widgets.upscale.selectQwen3VlEncoder',
    required: 'widgets.upscale.qwen3VlEncoderRequired',
  },
  qwenVlEncoderModel: {
    hint: 'qwenVlEncoder',
    label: 'widgets.upscale.qwenVlEncoder',
    modelTypes: ['qwen_vl_encoder'],
    placeholder: 'widgets.upscale.selectQwenVlEncoder',
    required: 'widgets.upscale.qwenVlEncoderRequired',
  },
  t5EncoderModel: {
    hint: 't5Encoder',
    label: 'widgets.upscale.t5Encoder',
    modelTypes: ['t5_encoder'],
    placeholder: 'widgets.upscale.selectT5Encoder',
    required: 'widgets.upscale.t5EncoderRequired',
  },
};

// Revision zero means the table is absent, whatever its request's status.
const selectCapabilitiesRevision = (snapshot: ArchitectureCapabilitiesSnapshot) => snapshot.revision;
const SCALE_MARKS = [1, 2, 4, 8, 16];
const CREATIVITY_MARKS = [UPSCALE_CREATIVITY_MIN, 0, UPSCALE_CREATIVITY_MAX];
const STRUCTURE_MARKS = [UPSCALE_STRUCTURE_MIN, 0, UPSCALE_STRUCTURE_MAX];
const TILE_SIZE_MARKS = [UPSCALE_TILE_SIZE_MIN, 1024, UPSCALE_TILE_SIZE_MAX];
const TILE_OVERLAP_MARKS = [UPSCALE_TILE_OVERLAP_MIN, 128, 256, UPSCALE_TILE_OVERLAP_MAX];

/** Scrub ranges cover everyday values; typing still reaches the validated bounds. */
const STEPS_SLIDER_MAX = 100;
const CFG_SLIDER_MAX = 20;
const ADVANCED_GRID_COLUMNS = { base: 1, md: 2 };
const PRESET_ENTRIES = Object.entries(UPSCALE_PRESETS);

const isSelectableMainModel = (model: ModelConfig): boolean => isSupportedUpscaleMainModel(model);

const UpscaleComponentPicker = memo(function UpscaleComponentPicker({
  onPatch,
  slot,
  value,
}: {
  onPatch: (patch: Partial<UpscaleWidgetValues>) => void;
  slot: UpscaleComponentSlot;
  value: UpscaleWidgetValues[UpscaleComponentField];
}) {
  const { t } = useTranslation();
  const ui = COMPONENT_FIELD_UI[slot.field];
  const select = useCallback(
    (model: ModelConfig | null) =>
      onPatch({ [slot.field]: isModelIdentifierConfig(model) && slot.accepts(model) ? model : null }),
    [onPatch, slot]
  );

  return (
    <Field error={value ? undefined : t(ui.required)} hint={ui.hint} label={t(ui.label)}>
      <ModelSelect
        filter={slot.accepts}
        invalid={!value}
        modelTypes={ui.modelTypes}
        placeholder={t(ui.placeholder)}
        value={value?.key ?? null}
        onChange={select}
      />
    </Field>
  );
});

const formatScale = (scale: number): string => `${scale}×`;

const getRangeError = (label: string, value: number, min: number, max: number): string | undefined =>
  Number.isFinite(value) && value >= min && value <= max ? undefined : `${label} must be between ${min} and ${max}.`;

/** The muted line under the preflight explaining how Model only reaches Scale with this upscaler. */
const useModelOnlyPlanNote = (nativeScale: number | null, scale: number): string | null => {
  const { t } = useTranslation();
  const plan = getModelOnlyUpscalePlan(nativeScale, scale);

  if (!plan || nativeScale === null) {
    return null;
  }
  if (plan.kind === 'no-enlarge') {
    return t('widgets.upscale.nativeScaleNoEnlarge');
  }
  if (nativeScale === scale) {
    return null;
  }

  return plan.resizes
    ? t('widgets.upscale.nativeScalePassesResize', { count: plan.passes, native: nativeScale, target: scale })
    : t('widgets.upscale.nativeScalePasses', { count: plan.passes, native: nativeScale });
};

const UpscaleOutputPreflight = memo(
  function UpscaleOutputPreflight({
    inputImage,
    mode,
    model,
    scale,
    upscaleModel,
  }: Pick<UpscaleWidgetValues, 'inputImage' | 'mode' | 'model' | 'scale' | 'upscaleModel'>) {
    const { t } = useTranslation();
    // Refine's detail pass resizes to Scale whatever the upscaler does, so only Model only explains the upscaler.
    const nativeScale = mode === 'model-only' ? getSpandrelNativeScale(upscaleModel) : null;
    const planNote = useModelOnlyPlanNote(nativeScale, scale);

    if (!inputImage) {
      return null;
    }

    const output = getUpscaleOutputDimensions(inputImage, { mode, model, scale, upscaleModel });
    const outputMegapixels = (output.width * output.height) / 1_000_000;
    const isLargeOutput = outputMegapixels >= LARGE_OUTPUT_MEGAPIXELS;

    return (
      <Stack bg="bg.subtle" gap="2" px="2.5" py="2" rounded="md">
        <DataList.Root gap="1.5" orientation="horizontal">
          <DataList.Item>
            <DataList.ItemLabel color="fg.subtle" fontSize="xs">
              {t('widgets.upscale.inputSize')}
            </DataList.ItemLabel>
            <DataList.ItemValue
              fontFamily="mono"
              fontSize="md"
              fontVariantNumeric="tabular-nums"
              justifyContent="flex-end"
            >
              {DIMENSION_FORMATTER.format(inputImage.width)} × {DIMENSION_FORMATTER.format(inputImage.height)}
            </DataList.ItemValue>
          </DataList.Item>
          <DataList.Item>
            <DataList.ItemLabel color="fg.subtle" fontSize="xs">
              {t('widgets.upscale.scale')}
            </DataList.ItemLabel>
            <DataList.ItemValue
              fontFamily="mono"
              fontSize="md"
              fontVariantNumeric="tabular-nums"
              justifyContent="flex-end"
            >
              {scale}×
            </DataList.ItemValue>
          </DataList.Item>
          {nativeScale === null ? null : (
            <DataList.Item>
              <DataList.ItemLabel color="fg.subtle" fontSize="xs">
                {t('widgets.upscale.nativeScale')}
              </DataList.ItemLabel>
              <DataList.ItemValue
                fontFamily="mono"
                fontSize="md"
                fontVariantNumeric="tabular-nums"
                justifyContent="flex-end"
              >
                {t('widgets.upscale.nativeScaleValue', { scale: nativeScale })}
              </DataList.ItemValue>
            </DataList.Item>
          )}
          <DataList.Item>
            <DataList.ItemLabel color="fg.subtle" fontSize="xs">
              {t('widgets.upscale.outputSize')}
            </DataList.ItemLabel>
            <DataList.ItemValue
              fontFamily="mono"
              fontSize="md"
              fontVariantNumeric="tabular-nums"
              fontWeight="semibold"
              justifyContent="flex-end"
            >
              {DIMENSION_FORMATTER.format(output.width)} × {DIMENSION_FORMATTER.format(output.height)}
            </DataList.ItemValue>
          </DataList.Item>
          <DataList.Item>
            <DataList.ItemLabel color="fg.subtle" fontSize="xs">
              {t('widgets.upscale.outputMegapixels')}
            </DataList.ItemLabel>
            <DataList.ItemValue
              fontFamily="mono"
              fontSize="md"
              fontVariantNumeric="tabular-nums"
              fontWeight="semibold"
              gap="1.5"
              justifyContent="flex-end"
            >
              {MEGAPIXEL_FORMATTER.format(outputMegapixels)} MP
              {isLargeOutput ? (
                <Badge colorPalette="orange" fontFamily="body" variant="surface">
                  {t('widgets.upscale.largeOutput')}
                </Badge>
              ) : null}
            </DataList.ItemValue>
          </DataList.Item>
        </DataList.Root>
        {planNote ? (
          <Text color="fg.muted" fontSize="xs" textWrap="pretty">
            {planNote}
          </Text>
        ) : null}
        {isLargeOutput ? (
          <Text
            borderTopWidth="1px"
            borderColor="border.subtle"
            color="fg.warning"
            fontSize="xs"
            pt="2"
            textWrap="pretty"
          >
            {t('widgets.upscale.largeOutputDescription')}
          </Text>
        ) : null}
      </Stack>
    );
  },
  (previous, next) =>
    previous.scale === next.scale &&
    previous.mode === next.mode &&
    // The grid comes from the architecture, so another model of the same base changes nothing.
    previous.model?.base === next.model?.base &&
    getSpandrelNativeScale(previous.upscaleModel) === getSpandrelNativeScale(next.upscaleModel) &&
    areInputImagesEquivalent(previous.inputImage, next.inputImage)
);

/** Re-identifying an upscaler records its native scale; shown under the field whose description says it is unknown. */
const OpenUpscalerInModelManager = ({ modelKey, onOpen }: { modelKey: string; onOpen: (modelKey: string) => void }) => {
  const { t } = useTranslation();
  const open = useCallback(() => onOpen(modelKey), [modelKey, onOpen]);

  return (
    <Button
      alignSelf="flex-start"
      fontSize="xs"
      h="auto"
      mt="-1.5"
      px="0"
      size="sm"
      textDecoration="underline"
      variant="plain"
      onClick={open}
    >
      {t('widgets.upscale.openInModelManager')}
    </Button>
  );
};

const UpscaleModelReconciler = ({
  rawValues,
  values,
}: {
  rawValues: Record<string, unknown>;
  values: UpscaleWidgetValues;
}) => {
  const { patchValues } = useUpscaleUiActions();

  useMountEffect(() => {
    const normalized = normalizeUpscaleWidgetValues(rawValues);

    if (normalized && valuesAreEqual(normalized, values)) {
      return;
    }

    patchValues({ ...values }, 'system');
  });

  return null;
};

export const UpscaleWidgetView = () => {
  const { t } = useTranslation();
  const openInModelManager = useOpenModelInManager();
  const conceptModels = useMemo<ConceptModelPort>(
    () => ({
      getBaseColorPalette: getModelBaseColorPalette,
      getBaseLabel: getModelBaseLabel,
      getImageUrl: getModelImageUrl,
      openInModelManager: openInModelManager ?? undefined,
    }),
    [openInModelManager]
  );
  const selection = useUpscaleUi();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const capabilitiesRevision = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    selectCapabilitiesRevision
  );
  // The sync picks VAEs and a new model's CFG from the served table, and a default applied without it
  // would stick: nothing re-applies one to a model already selected. So the sync waits for the table,
  // and until then the gate holds back the form, its model picker included.
  const hasCapabilities = capabilitiesRevision > 0;
  const canSync = modelsStatus === 'loaded' && hasCapabilities;
  const { patchPromptDraft: patchDraft, patchValues, projectId, promptDraft, rawValues } = selection;
  // Reuse normalization/reconciliation results across unrelated renders to preserve section identities.
  const values = useMemo(() => {
    const normalized = normalizeUpscaleWidgetValues(rawValues) ?? createDefaultUpscaleWidgetValues();

    return canSync ? syncUpscaleWidgetValuesWithModels(normalized, models) : normalized;
  }, [canSync, models, rawValues]);
  const modelsFingerprint = useMemo(
    () =>
      models
        .map(
          (model) =>
            `${model.key}:${model.hash}:${model.name}:${model.base}:${model.type}:${model.format}:${model.variant ?? ''}:${JSON.stringify(model.default_settings ?? null)}`
        )
        .join('|'),
    [models]
  );
  const errors = useMemo(
    () => ({
      cfgScale: getRangeError(t('widgets.upscale.cfgScale'), values.cfgScale, 1, upscaleMaxCfgScale(values.model)),
      creativity: getRangeError(
        t('widgets.upscale.creativity'),
        values.creativity,
        UPSCALE_CREATIVITY_MIN,
        UPSCALE_CREATIVITY_MAX
      ),
      scale: getRangeError(t('widgets.upscale.scale'), values.scale, UPSCALE_SCALE_MIN, UPSCALE_SCALE_MAX),
      seed: getRangeError(t('widgets.upscale.seed'), values.seed, 0, SEED_MAX),
      steps: getRangeError(t('widgets.upscale.steps'), values.steps, upscaleMinSteps(values.model), 1000),
      structure: getRangeError(
        t('widgets.upscale.structure'),
        values.structure,
        UPSCALE_STRUCTURE_MIN,
        UPSCALE_STRUCTURE_MAX
      ),
      tileOverlap: getRangeError(
        t('widgets.upscale.tileOverlap'),
        values.tileOverlap,
        UPSCALE_TILE_OVERLAP_MIN,
        UPSCALE_TILE_OVERLAP_MAX
      ),
      tileSize: getRangeError(
        t('widgets.upscale.tileSize'),
        values.tileSize,
        UPSCALE_TILE_SIZE_MIN,
        UPSCALE_TILE_SIZE_MAX
      ),
    }),
    [t, values]
  );
  const patch = useCallback((next: Partial<UpscaleWidgetValues>) => patchValues(next), [patchValues]);
  const patchPromptDraft = useCallback((next: ProjectPromptDraftPatch) => patchDraft(next), [patchDraft]);

  useMountEffect(() => {
    void ensureModelsLoaded();
    void ensureArchitectureCapabilitiesLoaded();
  });

  const selectMainModel = useCallback(
    (model: ModelConfig | null) => {
      if (!isMainModelConfig(model) || !isSelectableMainModel(model)) {
        return;
      }

      const nextValues = selectUpscaleMainModel(values, model as MainModelConfig, models);
      const notices: string[] = [];

      if (values.tileControlnetModel?.key !== nextValues.tileControlnetModel?.key) {
        notices.push(
          nextValues.tileControlnetModel
            ? t('widgets.upscale.controlNetChanged', { name: nextValues.tileControlnetModel.name })
            : t('widgets.upscale.controlNetCleared')
        );
      }
      if (values.vae && !nextValues.vae) {
        notices.push(t('widgets.upscale.vaeCleared'));
      }
      const removedLoraCount = values.loras.length - nextValues.loras.length;

      if (removedLoraCount > 0) {
        notices.push(t('widgets.upscale.lorasRemoved', { count: removedLoraCount }));
      }

      patch({ ...nextValues });

      if (notices.length > 0) {
        toaster.create({
          description: notices.join(' '),
          title: t('widgets.upscale.settingsAdjusted'),
          type: 'info',
        });
      }
    },
    [models, patch, t, values]
  );

  const addLora = useCallback(
    (model: ModelConfig | null) => {
      if (!isLoraModelConfig(model)) {
        return;
      }

      patchValues((current) =>
        current.model && isLoraCompatibleWithModel(model, current.model)
          ? { loras: [...current.loras, { isEnabled: true, model, weight: getDefaultLoraWeight(model) }] }
          : {}
      );
    },
    [patchValues]
  );
  const updateLora = useCallback(
    (key: string, update: Partial<GenerateLora>) =>
      // A row removed mid-edit still flushes its draft; that must not write an unchanged list.
      patchValues((current) =>
        current.loras.some((lora) => lora.model.key === key)
          ? { loras: current.loras.map((lora) => (lora.model.key === key ? { ...lora, ...update } : lora)) }
          : {}
      ),
    [patchValues]
  );
  const removeLora = useCallback(
    (key: string) =>
      patchValues((current) => ({ loras: current.loras.filter((candidate) => candidate.model.key !== key) })),
    [patchValues]
  );
  const selectedLoraKeys = useMemo(() => new Set(values.loras.map((lora) => lora.model.key)), [values.loras]);

  // A control with nothing to steer is worse than no control, so the widget follows the table
  // rather than showing every field for every architecture.
  const architecture = upscaleArchitectureFor(values.model);
  const tileControlNet = architecture?.tileControlNet ?? null;
  // Structure scales the ControlNet's weights, so with an optional one left out it steers nothing.
  const showStructure = tileControlNet !== null && (tileControlNet.required || values.tileControlnetModel !== null);
  const components = useMemo(() => upscaleComponentsFor(values.model), [values.model]);

  const activePresetId = useMemo(
    () =>
      PRESET_ENTRIES.find(
        ([, preset]) =>
          values.creativity === preset.creativity &&
          // Structure is hidden where nothing consumes it, so a stale value must not keep every
          // preset unhighlighted with nothing on screen to explain why.
          (!showStructure || values.structure === preset.structure)
      )?.[0] ?? null,
    [showStructure, values.creativity, values.structure]
  );
  const applyPreset = useCallback(
    ({ value }: { value: string | null }) => {
      const preset = value ? UPSCALE_PRESETS[value as keyof typeof UPSCALE_PRESETS] : undefined;

      if (preset) {
        patch({ creativity: preset.creativity, structure: preset.structure });
      }
    },
    [patch]
  );

  // Create field setters per patch identity so inline handlers cannot defeat section memoization.
  const set = useMemo(
    () => ({
      cfgScale: (cfgScale: number) => patch({ cfgScale }),
      clipSkip: (clipSkip: number) => patch({ clipSkip }),
      creativity: (creativity: number) => patch({ creativity }),
      inputImage: (inputImage: UpscaleWidgetValues['inputImage']) => patch({ inputImage }),
      mode: (mode: string) => {
        if (mode === 'refine' || mode === 'model-only') {
          patch({ mode });
        }
      },
      scale: (scale: number) => patch({ scale }),
      scheduler: (scheduler: string) => patch({ scheduler }),
      spandrelModel: (model: ModelConfig | null) =>
        patch({ upscaleModel: isSpandrelModelConfig(model) ? model : null }),
      steps: (steps: number) => patch({ steps }),
      structure: (structure: number) => patch({ structure }),
      tileOverlap: (tileOverlap: number) => patch({ tileOverlap }),
      tileSize: (tileSize: number) => patch({ tileSize }),
      vae: (model: ModelConfig | null) => patch({ vae: isVaeModelConfig(model) ? model : null }),
      vaePrecision: ({ value }: { value: string[] }) => {
        const vaePrecision = value[0];

        if (vaePrecision === 'fp16' || vaePrecision === 'fp32') {
          patch({ vaePrecision });
        }
      },
    }),
    [patch]
  );
  const loraFilter = useCallback(
    (model: ModelConfig) =>
      Boolean(values.model && isLoraModelConfig(model) && isLoraCompatibleWithModel(model, values.model)),
    [values.model]
  );
  const tileControlNetFilter = useCallback(
    (model: ModelConfig) => isTileControlNetCandidate(model, values.model),
    [values.model]
  );
  const setTileControlNet = useCallback(
    (model: ModelConfig | null) =>
      patch({ tileControlnetModel: isTileControlNetCandidate(model, values.model) ? model : null }),
    [patch, values.model]
  );
  // The served rule, not a base match: Z-Image decodes with a FLUX VAE, Anima with a Qwen-Image one.
  const vaeFilter = useCallback(
    (model: ModelConfig) =>
      Boolean(values.model && isVaeModelConfig(model) && isVaeCompatibleWithGenerateModel(values.model, model)),
    [values.model]
  );

  const vaePrecisionValue = useMemo(() => [values.vaePrecision], [values.vaePrecision]);
  // The architecture's own list, less the schedulers that would ignore Creativity, showing the one the
  // graph will really run. Where one is left there is nothing to choose, and the field goes.
  const schedulerOptions = useMemo(() => upscaleSchedulerOptions(values.model), [values.model]);
  const scheduler = coerceUpscaleScheduler(values.model, values.scheduler);
  const cfgScaleMax = upscaleMaxCfgScale(values.model);
  const gateLabels = useMemo(
    () => ({
      loadFailed: t('widgets.upscale.capabilitiesLoadFailed'),
      loading: t('widgets.upscale.loadingCapabilities'),
      retry: t('widgets.upscale.retry'),
    }),
    [t]
  );

  const sharedBadge = useMemo(() => <Badge fontFamily="mono">{t('widgets.upscale.shared')}</Badge>, [t]);
  const modeOptions = useMemo(
    () =>
      [
        { label: t('widgets.upscale.modeRefine'), value: 'refine' },
        { label: t('widgets.upscale.modeModelOnly'), value: 'model-only' },
      ] satisfies { label: string; value: UpscaleMode }[],
    [t]
  );
  // Model only keeps the Refine settings as they are, hidden, so switching back restores them.
  const isRefine = values.mode === 'refine';
  // Model only can neither show nor plan the passes of an upscaler installed before Invoke recorded native scales;
  // the field's description says so, and re-identifying it in the Model Manager records it.
  const isNativeScaleUnknown =
    values.mode === 'model-only' &&
    values.upscaleModel !== null &&
    getSpandrelNativeScale(values.upscaleModel) === null;

  const form = (
    <Stack gap={1} minW={0}>
      <UpscaleModelReconciler
        key={`${projectId}:${modelsStatus}:${capabilitiesRevision}:${modelsFingerprint}`}
        rawValues={rawValues}
        values={values}
      />

      <GenerationSettingsSection label={t('widgets.upscale.sourceAndTreatment')} defaultOpen>
        <Stack gap="3" p="2">
          <UpscaleImageField inputImage={values.inputImage} onChange={set.inputImage} />
          <UpscaleOutputPreflight
            inputImage={values.inputImage}
            mode={values.mode}
            model={values.model}
            scale={values.scale}
            upscaleModel={values.upscaleModel}
          />
          <Field
            error={values.upscaleModel ? undefined : t('widgets.upscale.spandrelModelRequired')}
            helpText={
              !values.upscaleModel
                ? undefined
                : isNativeScaleUnknown
                  ? t('widgets.upscale.nativeScaleUnknown')
                  : t('widgets.upscale.spandrelModelHelp')
            }
            hint="upscaleModel"
            label={t('widgets.upscale.spandrelModel')}
          >
            <ModelSelect
              invalid={!values.upscaleModel}
              modelTypes={SPANDREL_MODEL_TYPES}
              placeholder={t('widgets.upscale.selectSpandrelModel')}
              value={values.upscaleModel?.key ?? null}
              onChange={set.spandrelModel}
            />
          </Field>
          {isNativeScaleUnknown && values.upscaleModel && openInModelManager ? (
            <OpenUpscalerInModelManager modelKey={values.upscaleModel.key} onOpen={openInModelManager} />
          ) : null}
          <ScrubberField
            error={errors.scale}
            formatValue={formatScale}
            helpText={t('widgets.upscale.scaleHelp')}
            hint="upscaleScale"
            label={t('widgets.upscale.scale')}
            marks={SCALE_MARKS}
            max={UPSCALE_SCALE_MAX}
            min={UPSCALE_SCALE_MIN}
            step={0.5}
            value={values.scale}
            onChange={set.scale}
          />
          {isRefine ? (
            <SegmentGroup.Root
              aria-label={t('widgets.upscale.presetsLabel')}
              value={activePresetId}
              w="full"
              onValueChange={applyPreset}
            >
              <SegmentGroup.Indicator />
              {PRESET_ENTRIES.map(([id, preset]) => {
                const tooltipContent = `${t(`widgets.upscale.presetDescriptions.${id}`)} ${t(
                  'widgets.upscale.presetValues',
                  { creativity: preset.creativity, structure: preset.structure }
                )}`;

                return (
                  // The tooltip trigger merges onto the text, not the item: both
                  // tooltip and segment item write `data-state`, and the tooltip's
                  // open/closed would clobber the item's checked state.
                  <SegmentGroup.Item key={id} flex="1" minW="0" value={id}>
                    <SegmentGroup.ItemHiddenInput />
                    <Tooltip content={tooltipContent}>
                      <SegmentGroup.ItemText fontSize="md">{t(`widgets.upscale.presets.${id}`)}</SegmentGroup.ItemText>
                    </Tooltip>
                  </SegmentGroup.Item>
                );
              })}
            </SegmentGroup.Root>
          ) : null}
          {isRefine ? (
            <ScrubberField
              error={errors.creativity}
              helpText={t('widgets.upscale.creativityHelp')}
              hint="creativity"
              label={t('widgets.upscale.creativity')}
              marks={CREATIVITY_MARKS}
              max={UPSCALE_CREATIVITY_MAX}
              min={UPSCALE_CREATIVITY_MIN}
              step={1}
              value={values.creativity}
              onChange={set.creativity}
            />
          ) : null}
          {isRefine && showStructure && (
            <ScrubberField
              error={errors.structure}
              helpText={t('widgets.upscale.structureHelp')}
              hint="structure"
              label={t('widgets.upscale.structure')}
              marks={STRUCTURE_MARKS}
              max={UPSCALE_STRUCTURE_MAX}
              min={UPSCALE_STRUCTURE_MIN}
              step={1}
              value={values.structure}
              onChange={set.structure}
            />
          )}
        </Stack>
      </GenerationSettingsSection>

      {isRefine ? (
        <GenerationSettingsSection badges={sharedBadge} label={t('widgets.upscale.detailGuidance')}>
          <UpscalePromptFields
            loras={values.loras}
            model={values.model}
            negativePromptHeightPx={values.negativePromptHeightPx}
            positivePromptHeightPx={values.positivePromptHeightPx}
            promptDraft={promptDraft}
            projectId={projectId}
            showSyntaxHighlighting={selection.showPromptSyntaxHighlighting}
            onPatchPromptDraft={patchPromptDraft}
            onPatchValues={patch}
          />
        </GenerationSettingsSection>
      ) : null}

      {isRefine ? (
        <GenerationSettingsSection label={t('widgets.upscale.generation')}>
          <Stack gap="3" p="2">
            <Field
              error={values.model ? undefined : t('widgets.upscale.mainModelRequired')}
              hint="model"
              label={t('widgets.upscale.mainModel')}
            >
              <ModelSelect
                filter={isSelectableMainModel}
                invalid={!values.model}
                modelTypes={MAIN_MODEL_TYPES}
                placeholder={t('widgets.upscale.selectMainModel')}
                value={values.model?.key ?? null}
                onChange={selectMainModel}
              />
            </Field>
            <Separator borderColor="border.subtle" />
            {/* Iterations live in the top bar's invoke cluster, which edits this widget's batch count directly. */}
            <ScrubberField
              error={errors.steps}
              hint="steps"
              inputMax={1000}
              label={t('widgets.upscale.steps')}
              max={STEPS_SLIDER_MAX}
              min={upscaleMinSteps(values.model)}
              step={1}
              value={values.steps}
              onChange={set.steps}
            />
            <ScrubberField
              error={errors.cfgScale}
              hint="cfgScale"
              inputMax={cfgScaleMax}
              label={t('widgets.upscale.cfgScale')}
              max={Math.min(CFG_SLIDER_MAX, cfgScaleMax)}
              min={1}
              step={0.5}
              value={values.cfgScale}
              onChange={set.cfgScale}
            />
            {schedulerOptions.length > 1 ? (
              <Field hint="scheduler" label={t('widgets.upscale.scheduler')}>
                <Combobox
                  aria-label={t('widgets.upscale.scheduler')}
                  options={schedulerOptions}
                  value={scheduler}
                  onValueChange={set.scheduler}
                />
              </Field>
            ) : null}
            <Separator borderColor="border.subtle" />
            <SeedField
              batchCount={values.batchCount}
              error={errors.seed}
              label={t('widgets.upscale.seed')}
              seed={values.seed}
              seedMode={values.seedMode}
              onCommit={patch}
            />
            <Separator borderColor="border.subtle" />
            <Field hint="concepts" label={t('widgets.upscale.concepts')}>
              <ModelSelect
                excludeKeys={selectedLoraKeys}
                filter={loraFilter}
                modelTypes={LORA_MODEL_TYPES}
                placeholder={t('widgets.upscale.searchCompatibleConcepts')}
                scopeLabel={t('models.scopeConcepts')}
                value={null}
                onChange={addLora}
              />
            </Field>
            {values.loras.length > 0 ? (
              <Box mx={-1}>
                <ConceptList label={t('widgets.upscale.concepts')} projectId={projectId}>
                  {values.loras.map((lora) => (
                    <ConceptRow
                      key={lora.model.key}
                      models={conceptModels}
                      lora={lora}
                      onRemove={removeLora}
                      onUpdate={updateLora}
                    />
                  ))}
                </ConceptList>
              </Box>
            ) : null}
          </Stack>
        </GenerationSettingsSection>
      ) : null}

      {isRefine ? (
        <GenerationSettingsSection label={t('widgets.upscale.advanced')}>
          <Stack gap="3" p="2">
            {tileControlNet?.required && (
              <Field
                error={values.tileControlnetModel ? undefined : t('widgets.upscale.tileControlNetRequired')}
                helpText={values.tileControlnetModel ? t('widgets.upscale.tileControlNetHelp') : undefined}
                hint="tileControlNet"
                label={t('widgets.upscale.tileControlNet')}
              >
                <ModelSelect
                  filter={tileControlNetFilter}
                  invalid={!values.tileControlnetModel}
                  modelTypes={CONTROLNET_MODEL_TYPES}
                  placeholder={t('widgets.upscale.selectTileControlNet')}
                  value={values.tileControlnetModel?.key ?? null}
                  onChange={setTileControlNet}
                />
              </Field>
            )}
            {tileControlNet && !tileControlNet.required && (
              <Field
                helpText={t('widgets.upscale.tileControlNetOptionalHelp')}
                hint="tileControlNet"
                label={t('widgets.upscale.tileControlNet')}
              >
                <ModelSelect
                  filter={tileControlNetFilter}
                  isClearable
                  modelTypes={CONTROLNET_MODEL_TYPES}
                  placeholder={t('widgets.upscale.selectOptionalTileControlNet')}
                  value={values.tileControlnetModel?.key ?? null}
                  onChange={setTileControlNet}
                />
              </Field>
            )}
            {components.slots.map((slot) => (
              <UpscaleComponentPicker key={slot.field} slot={slot} value={values[slot.field]} onPatch={patch} />
            ))}
            <ScrubberField
              error={errors.tileSize}
              helpText={t('widgets.upscale.tileSizeHelp')}
              hint="tileSize"
              label={t('widgets.upscale.tileSize')}
              marks={TILE_SIZE_MARKS}
              max={UPSCALE_TILE_SIZE_MAX}
              min={UPSCALE_TILE_SIZE_MIN}
              step={64}
              value={values.tileSize}
              onChange={set.tileSize}
            />
            <ScrubberField
              error={errors.tileOverlap}
              helpText={t('widgets.upscale.tileOverlapHelp')}
              hint="tileOverlap"
              label={t('widgets.upscale.tileOverlap')}
              marks={TILE_OVERLAP_MARKS}
              max={UPSCALE_TILE_OVERLAP_MAX}
              min={UPSCALE_TILE_OVERLAP_MIN}
              step={8}
              value={values.tileOverlap}
              onChange={set.tileOverlap}
            />
            <SimpleGrid columns={ADVANCED_GRID_COLUMNS} gap="2">
              <Field
                error={components.vae && !values.vae ? t('widgets.upscale.vaeRequired') : undefined}
                hint="vae"
                label={t('widgets.upscale.vae')}
              >
                <ModelSelect
                  filter={vaeFilter}
                  invalid={components.vae && !values.vae}
                  isClearable={!components.vae}
                  modelTypes={VAE_MODEL_TYPES}
                  placeholder={components.vae ? t('widgets.upscale.selectVae') : t('widgets.upscale.bundledVae')}
                  value={values.vae?.key ?? null}
                  onChange={set.vae}
                />
              </Field>
              <Field hint="vaePrecision" label={t('widgets.upscale.vaePrecision')}>
                <Select
                  aria-label={t('widgets.upscale.vaePrecision')}
                  collection={VAE_PRECISION_COLLECTION}
                  value={vaePrecisionValue}
                  onValueChange={set.vaePrecision}
                />
              </Field>
            </SimpleGrid>
            {values.model?.base === 'sd-1' ? (
              <ScrubberField
                hint="clipSkip"
                label={t('widgets.upscale.clipSkip')}
                max={12}
                min={0}
                step={1}
                value={values.clipSkip}
                onChange={set.clipSkip}
              />
            ) : null}
          </Stack>
        </GenerationSettingsSection>
      ) : null}
    </Stack>
  );

  // The mode sits above the gate: Model only reads no main-model capabilities, so it stays reachable and usable while
  // the table is loading or has failed, which holds back only Refine's form.
  return (
    <Stack gap={1} minW={0} p="1">
      <Box px="2" pt="2">
        <Field helpText={t('widgets.upscale.modeHelp')} label={t('widgets.upscale.mode')}>
          <SegmentedControl
            ariaLabel={t('widgets.upscale.mode')}
            options={modeOptions}
            value={values.mode}
            onChange={set.mode}
          />
        </Field>
      </Box>
      <CapabilitiesGate isReady={hasCapabilities || !isRefine} labels={gateLabels}>
        {form}
      </CapabilitiesGate>
    </Stack>
  );
};

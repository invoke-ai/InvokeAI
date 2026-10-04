import type { GenerateLora, MainModelConfig } from '@features/generation/contracts';
import type { ProjectPromptDraftPatch } from '@features/generation/settings';
import type { ModelConfig, ModelTaxonomyType } from '@features/models';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';

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
  ConceptList,
  type ConceptModelPort,
  ConceptRow,
  GenerationSettingsSection,
  SeedField,
} from '@features/generation/components';
import {
  getDefaultLoraWeight,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  isMainModelConfig,
  isModelIdentifierConfig,
  isVaeModelConfig,
  SCHEDULER_OPTIONS,
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
  createDefaultUpscaleWidgetValues,
  needsExplicitComponents,
  getUpscaleOutputDimensions,
  isSpandrelModelConfig,
  isSupportedUpscaleMainModel,
  isTileControlNetCandidate,
  normalizeUpscaleWidgetValues,
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
} from '@features/upscale/core/settings';
import { SEED_MAX } from '@platform/core/seed';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Combobox, Field, Select, Tooltip } from '@platform/ui';
import { ScrubberField } from '@platform/ui/ScrubberField';
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
const T5_ENCODER_MODEL_TYPES: readonly ModelTaxonomyType[] = ['t5_encoder'];
const CLIP_EMBED_MODEL_TYPES: readonly ModelTaxonomyType[] = ['clip_embed'];

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

const formatScale = (scale: number): string => `${scale}×`;

const getRangeError = (label: string, value: number, min: number, max: number): string | undefined =>
  Number.isFinite(value) && value >= min && value <= max ? undefined : `${label} must be between ${min} and ${max}.`;

const UpscaleOutputPreflight = memo(
  function UpscaleOutputPreflight({
    inputImage,
    scale,
  }: {
    inputImage: UpscaleWidgetValues['inputImage'];
    scale: number;
  }) {
    const { t } = useTranslation();

    if (!inputImage) {
      return null;
    }

    const output = getUpscaleOutputDimensions(inputImage, scale);
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
  (previous, next) => previous.scale === next.scale && areInputImagesEquivalent(previous.inputImage, next.inputImage)
);

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
  const { patchPromptDraft: patchDraft, patchValues, projectId, promptDraft, rawValues } = selection;
  // Reuse normalization/reconciliation results across unrelated renders to preserve section identities.
  const values = useMemo(() => {
    const normalized = normalizeUpscaleWidgetValues(rawValues) ?? createDefaultUpscaleWidgetValues();

    return modelsStatus === 'loaded' ? syncUpscaleWidgetValuesWithModels(normalized, models) : normalized;
  }, [models, modelsStatus, rawValues]);
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
      cfgScale: getRangeError(t('widgets.upscale.cfgScale'), values.cfgScale, 1, 100),
      creativity: getRangeError(
        t('widgets.upscale.creativity'),
        values.creativity,
        UPSCALE_CREATIVITY_MIN,
        UPSCALE_CREATIVITY_MAX
      ),
      scale: getRangeError(t('widgets.upscale.scale'), values.scale, UPSCALE_SCALE_MIN, UPSCALE_SCALE_MAX),
      seed: getRangeError(t('widgets.upscale.seed'), values.seed, 0, SEED_MAX),
      steps: getRangeError(t('widgets.upscale.steps'), values.steps, 1, 1000),
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
  });

  const selectMainModel = useCallback(
    (model: ModelConfig | null) => {
      if (!isMainModelConfig(model) || !isSelectableMainModel(model)) {
        return;
      }

      const nextValues = syncUpscaleWidgetValuesWithModels({ ...values, model: model as MainModelConfig }, models);
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
  const showTileControlNet = architecture?.usesTileControlNet ?? false;
  const showComponentPickers = needsExplicitComponents(values.model);

  const activePresetId = useMemo(
    () =>
      PRESET_ENTRIES.find(
        ([, preset]) =>
          values.creativity === preset.creativity &&
          // Structure is hidden where nothing consumes it, so a stale value must not keep every
          // preset unhighlighted with nothing on screen to explain why.
          (!showTileControlNet || values.structure === preset.structure)
      )?.[0] ?? null,
    [showTileControlNet, values.creativity, values.structure]
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
  const vaeFilter = useCallback(
    (model: ModelConfig) => Boolean(values.model && model.base === values.model.base),
    [values.model]
  );

  const vaePrecisionValue = useMemo(() => [values.vaePrecision], [values.vaePrecision]);

  const setT5Encoder = useCallback(
    (model: ModelConfig | null) => patch({ t5EncoderModel: isModelIdentifierConfig(model) ? model : null }),
    [patch]
  );
  const setClipEmbed = useCallback(
    (model: ModelConfig | null) => patch({ clipEmbedModel: isModelIdentifierConfig(model) ? model : null }),
    [patch]
  );

  const sharedBadge = useMemo(() => <Badge fontFamily="mono">{t('widgets.upscale.shared')}</Badge>, [t]);

  return (
    <Stack gap={1} minW={0} p="1">
      <UpscaleModelReconciler
        key={`${projectId}:${modelsStatus}:${modelsFingerprint}`}
        rawValues={rawValues}
        values={values}
      />

      <GenerationSettingsSection label={t('widgets.upscale.sourceAndTreatment')} defaultOpen>
        <Stack gap="3" p="2">
          <UpscaleImageField inputImage={values.inputImage} onChange={set.inputImage} />
          <UpscaleOutputPreflight inputImage={values.inputImage} scale={values.scale} />
          <Field
            error={values.upscaleModel ? undefined : t('widgets.upscale.spandrelModelRequired')}
            helpText={values.upscaleModel ? t('widgets.upscale.spandrelModelHelp') : undefined}
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
          {showTileControlNet && (
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
            min={1}
            step={1}
            value={values.steps}
            onChange={set.steps}
          />
          <ScrubberField
            error={errors.cfgScale}
            hint="cfgScale"
            inputMax={100}
            label={t('widgets.upscale.cfgScale')}
            max={CFG_SLIDER_MAX}
            min={1}
            step={0.5}
            value={values.cfgScale}
            onChange={set.cfgScale}
          />
          <Field hint="scheduler" label={t('widgets.upscale.scheduler')}>
            <Combobox
              aria-label={t('widgets.upscale.scheduler')}
              options={SCHEDULER_OPTIONS}
              value={values.scheduler}
              onValueChange={set.scheduler}
            />
          </Field>
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

      <GenerationSettingsSection label={t('widgets.upscale.advanced')}>
        <Stack gap="3" p="2">
          {showTileControlNet && (
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
          {showComponentPickers && (
            <>
              <Field
                error={values.t5EncoderModel ? undefined : t('widgets.upscale.t5EncoderRequired')}
                hint="t5Encoder"
                label={t('widgets.upscale.t5Encoder')}
              >
                <ModelSelect
                  invalid={!values.t5EncoderModel}
                  modelTypes={T5_ENCODER_MODEL_TYPES}
                  placeholder={t('widgets.upscale.selectT5Encoder')}
                  value={values.t5EncoderModel?.key ?? null}
                  onChange={setT5Encoder}
                />
              </Field>
              <Field
                error={values.clipEmbedModel ? undefined : t('widgets.upscale.clipEmbedRequired')}
                hint="clipEmbed"
                label={t('widgets.upscale.clipEmbed')}
              >
                <ModelSelect
                  invalid={!values.clipEmbedModel}
                  modelTypes={CLIP_EMBED_MODEL_TYPES}
                  placeholder={t('widgets.upscale.selectClipEmbed')}
                  value={values.clipEmbedModel?.key ?? null}
                  onChange={setClipEmbed}
                />
              </Field>
            </>
          )}
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
              error={showComponentPickers && !values.vae ? t('widgets.upscale.vaeRequired') : undefined}
              hint="vae"
              label={t('widgets.upscale.vae')}
            >
              <ModelSelect
                filter={vaeFilter}
                invalid={showComponentPickers && !values.vae}
                isClearable={!showComponentPickers}
                modelTypes={VAE_MODEL_TYPES}
                placeholder={showComponentPickers ? t('widgets.upscale.selectVae') : t('widgets.upscale.bundledVae')}
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
    </Stack>
  );
};

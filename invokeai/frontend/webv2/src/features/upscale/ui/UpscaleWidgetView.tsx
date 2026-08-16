import type { GenerateLora, MainModelConfig } from '@features/generation/contracts';
import type { ProjectPromptDraftPatch } from '@features/generation/settings';
import type { ModelConfig, ModelTaxonomyType } from '@features/models';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';

import {
  Badge,
  createListCollection,
  DataList,
  HStack,
  NumberInput,
  SegmentGroup,
  SimpleGrid,
  Stack,
  Switch,
  Text,
} from '@chakra-ui/react';
import { GenerationSettingsSection } from '@features/generation/components';
import {
  SCHEDULER_OPTIONS,
  getDefaultLoraWeight,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  isMainModelConfig,
  isVaeModelConfig,
  SEED_MAX,
} from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { ModelSelect } from '@features/models/react';
import {
  createDefaultUpscaleWidgetValues,
  getUpscaleOutputDimensions,
  isSpandrelModelConfig,
  isSupportedUpscaleMainModel,
  isTileControlNetCandidate,
  normalizeUpscaleWidgetValues,
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
import { useMountEffect } from '@platform/react/useMountEffect';
import { Combobox, Field, IconButton, Select, Tooltip } from '@platform/ui';
import { SliderNumberField } from '@platform/ui/SliderNumberField';
import { toaster } from '@platform/ui/toaster';
import { DicesIcon } from 'lucide-react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { areInputImagesEquivalent, valuesAreEqual } from './upscaleComparators';
import { UpscaleLoraRow, UpscalePromptFields } from './UpscaleFormFields';
import { UpscaleImageField } from './UpscaleImageField';
import { useUpscaleUi, useUpscaleUiActions } from './UpscaleUiContext';

/**
 * Every prop identity in this file is stable by construction — module-scope
 * constants for literals, `useCallback`/`useMemo` for anything closing over
 * state, and `memo` on each section. The widget re-renders on every keystroke
 * that patches project state, so an inline `{...}`/`() => …` prop anywhere here
 * re-renders the whole form (prompt editors and model pickers included) for a
 * change that touched one number. The `react-perf` lint rules enforce this;
 * they were previously disabled file-wide.
 */

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

const SCALE_MARKS = [1, 2, 4, 8, 16];
const CREATIVITY_MARKS = [UPSCALE_CREATIVITY_MIN, 0, UPSCALE_CREATIVITY_MAX];
const STRUCTURE_MARKS = [UPSCALE_STRUCTURE_MIN, 0, UPSCALE_STRUCTURE_MAX];
const TILE_SIZE_MARKS = [UPSCALE_TILE_SIZE_MIN, 1024, UPSCALE_TILE_SIZE_MAX];
const TILE_OVERLAP_MARKS = [UPSCALE_TILE_OVERLAP_MIN, 128, 256, UPSCALE_TILE_OVERLAP_MAX];

const GENERATION_GRID_COLUMNS = { base: 2, md: 3 };
const ADVANCED_GRID_COLUMNS = { base: 1, md: 2 };
const SWITCH_CHECKED_PROPS = { bg: 'accent.solid' };
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
        <DataList.Root gap="1.5" orientation="horizontal" size="sm">
          <DataList.Item>
            <DataList.ItemLabel color="fg.subtle" fontSize="2xs">
              {t('widgets.upscale.inputSize')}
            </DataList.ItemLabel>
            <DataList.ItemValue
              fontFamily="mono"
              fontSize="xs"
              fontVariantNumeric="tabular-nums"
              justifyContent="flex-end"
            >
              {DIMENSION_FORMATTER.format(inputImage.width)} × {DIMENSION_FORMATTER.format(inputImage.height)}
            </DataList.ItemValue>
          </DataList.Item>
          <DataList.Item>
            <DataList.ItemLabel color="fg.subtle" fontSize="2xs">
              {t('widgets.upscale.scale')}
            </DataList.ItemLabel>
            <DataList.ItemValue
              fontFamily="mono"
              fontSize="xs"
              fontVariantNumeric="tabular-nums"
              justifyContent="flex-end"
            >
              {scale}×
            </DataList.ItemValue>
          </DataList.Item>
          <DataList.Item>
            <DataList.ItemLabel color="fg.subtle" fontSize="2xs">
              {t('widgets.upscale.outputSize')}
            </DataList.ItemLabel>
            <DataList.ItemValue
              fontFamily="mono"
              fontSize="xs"
              fontVariantNumeric="tabular-nums"
              fontWeight="semibold"
              justifyContent="flex-end"
            >
              {DIMENSION_FORMATTER.format(output.width)} × {DIMENSION_FORMATTER.format(output.height)}
            </DataList.ItemValue>
          </DataList.Item>
          <DataList.Item>
            <DataList.ItemLabel color="fg.subtle" fontSize="2xs">
              {t('widgets.upscale.outputMegapixels')}
            </DataList.ItemLabel>
            <DataList.ItemValue
              fontFamily="mono"
              fontSize="xs"
              fontVariantNumeric="tabular-nums"
              fontWeight="semibold"
              gap="1.5"
              justifyContent="flex-end"
            >
              {MEGAPIXEL_FORMATTER.format(outputMegapixels)} MP
              {isLargeOutput ? (
                <Badge colorPalette="orange" fontFamily="body" size="xs" variant="surface">
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
            fontSize="2xs"
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
  const selection = useUpscaleUi();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const { patchPromptDraft: patchDraft, patchValues, projectId, promptDraft, rawValues } = selection;
  // Normalizing and reconciling against the model list is the widget's most
  // expensive derivation; it must not run on unrelated re-renders, and a fresh
  // `values` identity would re-render every section below.
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
      cfgScale: getRangeError(t('widgets.upscale.cfgScale'), values.cfgScale, 0, 100),
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
      if (!values.model || !isLoraModelConfig(model) || !isLoraCompatibleWithModel(model, values.model)) {
        return;
      }

      patch({ loras: [...values.loras, { isEnabled: true, model, weight: getDefaultLoraWeight(model) }] });
    },
    [patch, values.loras, values.model]
  );
  const updateLora = useCallback(
    (key: string, update: Partial<GenerateLora>) =>
      patch({ loras: values.loras.map((lora) => (lora.model.key === key ? { ...lora, ...update } : lora)) }),
    [patch, values.loras]
  );
  const removeLora = useCallback(
    (key: string) => patch({ loras: values.loras.filter((candidate) => candidate.model.key !== key) }),
    [patch, values.loras]
  );
  const selectedLoraKeys = useMemo(() => new Set(values.loras.map((lora) => lora.model.key)), [values.loras]);

  const activePresetId = useMemo(
    () =>
      PRESET_ENTRIES.find(
        ([, preset]) => values.creativity === preset.creativity && values.structure === preset.structure
      )?.[0] ?? null,
    [values.creativity, values.structure]
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

  // One setter per field, created once per `patch` identity: inline
  // `onChange={(x) => patch({ x })}` props would defeat every `memo` below.
  const set = useMemo(
    () => ({
      batchCount: ({ valueAsNumber }: NumberInput.ValueChangeDetails) =>
        Number.isFinite(valueAsNumber) && patch({ batchCount: valueAsNumber }),
      cfgScale: ({ valueAsNumber }: NumberInput.ValueChangeDetails) =>
        Number.isFinite(valueAsNumber) && patch({ cfgScale: valueAsNumber }),
      clipSkip: ({ valueAsNumber }: NumberInput.ValueChangeDetails) =>
        Number.isFinite(valueAsNumber) && patch({ clipSkip: valueAsNumber }),
      creativity: (creativity: number) => patch({ creativity }),
      inputImage: (inputImage: UpscaleWidgetValues['inputImage']) => patch({ inputImage }),
      randomizeSeed: (details: { checked: boolean }) => patch({ shouldRandomizeSeed: details.checked }),
      scale: (scale: number) => patch({ scale }),
      scheduler: (scheduler: string) => patch({ scheduler }),
      seed: ({ valueAsNumber }: NumberInput.ValueChangeDetails) =>
        Number.isFinite(valueAsNumber) && patch({ seed: valueAsNumber }),
      shuffleSeed: () => patch({ seed: Math.floor(Math.random() * SEED_MAX) }),
      spandrelModel: (model: ModelConfig | null) =>
        patch({ upscaleModel: isSpandrelModelConfig(model) ? model : null }),
      steps: ({ valueAsNumber }: NumberInput.ValueChangeDetails) =>
        Number.isFinite(valueAsNumber) && patch({ steps: valueAsNumber }),
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
  // Model filters close over the selected main model, so they change only when
  // that model does — not on every keystroke elsewhere in the form.
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

  const sharedBadge = useMemo(
    () => (
      <Badge fontFamily="mono" size="xs">
        {t('widgets.upscale.shared')}
      </Badge>
    ),
    [t]
  );

  return (
    <Stack gap="1" minW="0" p="1">
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
              size="xs"
              value={values.upscaleModel?.key ?? null}
              onChange={set.spandrelModel}
            />
          </Field>
          <Field
            error={errors.scale}
            helpText={t('widgets.upscale.scaleHelp')}
            hint="upscaleScale"
            label={t('widgets.upscale.scale')}
          >
            <SliderNumberField
              ariaLabel={t('widgets.upscale.scale')}
              formatValue={formatScale}
              marks={SCALE_MARKS}
              max={UPSCALE_SCALE_MAX}
              min={UPSCALE_SCALE_MIN}
              showStepper
              step={0.5}
              value={values.scale}
              onChange={set.scale}
            />
          </Field>
          <SegmentGroup.Root
            aria-label={t('widgets.upscale.presetsLabel')}
            size="xs"
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
                    <SegmentGroup.ItemText fontSize="xs">{t(`widgets.upscale.presets.${id}`)}</SegmentGroup.ItemText>
                  </Tooltip>
                </SegmentGroup.Item>
              );
            })}
          </SegmentGroup.Root>
          <Field
            error={errors.creativity}
            helpText={t('widgets.upscale.creativityHelp')}
            hint="creativity"
            label={t('widgets.upscale.creativity')}
          >
            <SliderNumberField
              ariaLabel={t('widgets.upscale.creativity')}
              marks={CREATIVITY_MARKS}
              max={UPSCALE_CREATIVITY_MAX}
              min={UPSCALE_CREATIVITY_MIN}
              showStepper
              step={1}
              value={values.creativity}
              onChange={set.creativity}
            />
          </Field>
          <Field
            error={errors.structure}
            helpText={t('widgets.upscale.structureHelp')}
            hint="structure"
            label={t('widgets.upscale.structure')}
          >
            <SliderNumberField
              ariaLabel={t('widgets.upscale.structure')}
              marks={STRUCTURE_MARKS}
              max={UPSCALE_STRUCTURE_MAX}
              min={UPSCALE_STRUCTURE_MIN}
              showStepper
              step={1}
              value={values.structure}
              onChange={set.structure}
            />
          </Field>
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
              size="xs"
              value={values.model?.key ?? null}
              onChange={selectMainModel}
            />
          </Field>
          <SimpleGrid columns={GENERATION_GRID_COLUMNS} gap="2">
            <Field error={errors.steps} hint="steps" label={t('widgets.upscale.steps')}>
              <NumberInput.Root max={1000} min={1} size="xs" value={String(values.steps)} onValueChange={set.steps}>
                <NumberInput.Control />
                <NumberInput.Input fontVariantNumeric="tabular-nums" />
              </NumberInput.Root>
            </Field>
            <Field error={errors.cfgScale} hint="cfgScale" label={t('widgets.upscale.cfgScale')}>
              <NumberInput.Root
                max={100}
                min={0}
                size="xs"
                step={0.5}
                value={String(values.cfgScale)}
                onValueChange={set.cfgScale}
              >
                <NumberInput.Control />
                <NumberInput.Input fontVariantNumeric="tabular-nums" />
              </NumberInput.Root>
            </Field>
            <Field hint="batchCount" label={t('widgets.upscale.batchCount')}>
              <NumberInput.Root min={1} size="xs" value={String(values.batchCount)} onValueChange={set.batchCount}>
                <NumberInput.Control />
                <NumberInput.Input fontVariantNumeric="tabular-nums" />
              </NumberInput.Root>
            </Field>
          </SimpleGrid>
          <Field hint="scheduler" label={t('widgets.upscale.scheduler')}>
            <Combobox
              aria-label={t('widgets.upscale.scheduler')}
              options={SCHEDULER_OPTIONS}
              size="xs"
              value={values.scheduler}
              onValueChange={set.scheduler}
            />
          </Field>
          <Field
            error={values.shouldRandomizeSeed ? undefined : errors.seed}
            hint="seed"
            label={t('widgets.upscale.seed')}
          >
            <HStack gap="2">
              <NumberInput.Root
                disabled={values.shouldRandomizeSeed}
                max={SEED_MAX}
                min={0}
                size="xs"
                value={String(values.seed)}
                w="full"
                onValueChange={set.seed}
              >
                <NumberInput.Input fontVariantNumeric="tabular-nums" />
              </NumberInput.Root>
              <Tooltip content={t('widgets.upscale.shuffleSeed')}>
                <IconButton
                  aria-label={t('widgets.upscale.shuffleSeed')}
                  disabled={values.shouldRandomizeSeed}
                  size="xs"
                  variant="outline"
                  onClick={set.shuffleSeed}
                >
                  <DicesIcon />
                </IconButton>
              </Tooltip>
              <Switch.Root checked={values.shouldRandomizeSeed} size="sm" onCheckedChange={set.randomizeSeed}>
                <Switch.HiddenInput />
                <Switch.Control _checked={SWITCH_CHECKED_PROPS}>
                  <Switch.Thumb />
                </Switch.Control>
                <Switch.Label fontSize="xs">{t('widgets.upscale.random')}</Switch.Label>
              </Switch.Root>
            </HStack>
          </Field>
          <Field hint="concepts" label={t('widgets.upscale.addLora')}>
            <ModelSelect
              excludeKeys={selectedLoraKeys}
              filter={loraFilter}
              modelTypes={LORA_MODEL_TYPES}
              placeholder={t('widgets.upscale.selectLora')}
              size="xs"
              value={null}
              onChange={addLora}
            />
          </Field>
          {values.loras.map((lora) => (
            <UpscaleLoraRow key={lora.model.key} lora={lora} onRemove={removeLora} onUpdate={updateLora} />
          ))}
        </Stack>
      </GenerationSettingsSection>

      <GenerationSettingsSection label={t('widgets.upscale.advanced')}>
        <Stack gap="3" p="2">
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
              size="xs"
              value={values.tileControlnetModel?.key ?? null}
              onChange={setTileControlNet}
            />
          </Field>
          <Field
            error={errors.tileSize}
            helpText={t('widgets.upscale.tileSizeHelp')}
            hint="tileSize"
            label={t('widgets.upscale.tileSize')}
          >
            <SliderNumberField
              ariaLabel={t('widgets.upscale.tileSize')}
              marks={TILE_SIZE_MARKS}
              max={UPSCALE_TILE_SIZE_MAX}
              min={UPSCALE_TILE_SIZE_MIN}
              showStepper
              step={64}
              value={values.tileSize}
              onChange={set.tileSize}
            />
          </Field>
          <Field
            error={errors.tileOverlap}
            helpText={t('widgets.upscale.tileOverlapHelp')}
            hint="tileOverlap"
            label={t('widgets.upscale.tileOverlap')}
          >
            <SliderNumberField
              ariaLabel={t('widgets.upscale.tileOverlap')}
              marks={TILE_OVERLAP_MARKS}
              max={UPSCALE_TILE_OVERLAP_MAX}
              min={UPSCALE_TILE_OVERLAP_MIN}
              showStepper
              step={8}
              value={values.tileOverlap}
              onChange={set.tileOverlap}
            />
          </Field>
          <SimpleGrid columns={ADVANCED_GRID_COLUMNS} gap="2">
            <Field
              hint="vae"
              label={t('widgets.upscale.vae')}
              helpText={values.vae ? undefined : t('widgets.upscale.bundledVae')}
            >
              <ModelSelect
                filter={vaeFilter}
                isClearable
                modelTypes={VAE_MODEL_TYPES}
                placeholder={t('widgets.upscale.bundledVae')}
                size="xs"
                value={values.vae?.key ?? null}
                onChange={set.vae}
              />
            </Field>
            <Field hint="vaePrecision" label={t('widgets.upscale.vaePrecision')}>
              <Select
                aria-label={t('widgets.upscale.vaePrecision')}
                collection={VAE_PRECISION_COLLECTION}
                size="xs"
                value={vaePrecisionValue}
                onValueChange={set.vaePrecision}
              />
            </Field>
          </SimpleGrid>
          {values.model?.base === 'sd-1' ? (
            <Field hint="clipSkip" label={t('widgets.upscale.clipSkip')}>
              <NumberInput.Root max={12} min={0} size="xs" value={String(values.clipSkip)} onValueChange={set.clipSkip}>
                <NumberInput.Control />
                <NumberInput.Input fontVariantNumeric="tabular-nums" />
              </NumberInput.Root>
            </Field>
          ) : null}
        </Stack>
      </GenerationSettingsSection>
    </Stack>
  );
};

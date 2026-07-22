/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { GenerateLora, MainModelConfig, PromptHistoryItem } from '@features/generation/contracts';
import type { ProjectPromptDraft, ProjectPromptDraftPatch } from '@features/generation/settings';
import type { ModelConfig } from '@features/models';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';
import type { ChangeEvent } from 'react';

import {
  Badge,
  Box,
  ButtonGroup,
  createListCollection,
  DataList,
  HStack,
  Image,
  Input,
  NumberInput,
  SimpleGrid,
  Spinner,
  Stack,
  Switch,
  Text,
} from '@chakra-ui/react';
import { useDndMonitor, useDroppable } from '@dnd-kit/core';
import { galleryImages, galleryTransfers } from '@features/gallery';
import { galleryImageUrls, isGalleryImageDragData } from '@features/gallery/utility';
import { GenerationSettingsSection, NegativePromptField, PositivePromptField } from '@features/generation/components';
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
import { Button, Combobox, DropZone, Field, IconButton, Select, Slider, Tooltip } from '@platform/ui';
import { toaster } from '@platform/ui/toaster';
import { DicesIcon, ImagePlusIcon, Trash2Icon, UploadIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useUpscaleUi } from './UpscaleUiContext';

const DROP_ID = 'upscale-input-image';
const VAE_PRECISION_COLLECTION = createListCollection({
  items: [
    { label: 'FP16', value: 'fp16' },
    { label: 'FP32', value: 'fp32' },
  ] as const,
});
const LARGE_OUTPUT_MEGAPIXELS = 50;
const DIMENSION_FORMATTER = new Intl.NumberFormat();
const MEGAPIXEL_FORMATTER = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 });

const isSelectableMainModel = (model: ModelConfig): boolean => isSupportedUpscaleMainModel(model);

const valuesAreEqual = (left: UpscaleWidgetValues, right: UpscaleWidgetValues): boolean =>
  JSON.stringify(left) === JSON.stringify(right);

const getRangeError = (label: string, value: number, min: number, max: number): string | undefined =>
  Number.isFinite(value) && value >= min && value <= max ? undefined : `${label} must be between ${min} and ${max}.`;

const UpscaleOutputPreflight = ({ values }: { values: UpscaleWidgetValues }) => {
  const { t } = useTranslation();

  if (!values.inputImage) {
    return null;
  }

  const output = getUpscaleOutputDimensions(values.inputImage, values.scale);
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
            {DIMENSION_FORMATTER.format(values.inputImage.width)} ×{' '}
            {DIMENSION_FORMATTER.format(values.inputImage.height)}
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
            {values.scale}×
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
};

const UpscaleModelReconciler = ({
  rawValues,
  values,
}: {
  rawValues: Record<string, unknown>;
  values: UpscaleWidgetValues;
}) => {
  const { patchValues } = useUpscaleUi();

  useMountEffect(() => {
    const normalized = normalizeUpscaleWidgetValues(rawValues);

    if (normalized && valuesAreEqual(normalized, values)) {
      return;
    }

    patchValues({ ...values });
  });

  return null;
};

const UpscalePromptFields = ({
  onPatchPromptDraft,
  onPatchValues,
  promptDraft,
  projectId,
  showSyntaxHighlighting,
  values,
}: {
  onPatchPromptDraft: (patch: ProjectPromptDraftPatch) => void;
  onPatchValues: (patch: Partial<UpscaleWidgetValues>) => void;
  promptDraft: ProjectPromptDraft;
  projectId: string;
  showSyntaxHighlighting: boolean;
  values: UpscaleWidgetValues;
}) => {
  const { t } = useTranslation();
  const handleUsePrompt = useCallback(
    (prompt: PromptHistoryItem) =>
      onPatchPromptDraft({
        negativePrompt: prompt.negativePrompt ?? '',
        negativePromptEnabled: prompt.negativePrompt ? true : promptDraft.negativePromptEnabled,
        positivePrompt: prompt.positivePrompt,
      }),
    [onPatchPromptDraft, promptDraft.negativePromptEnabled]
  );

  return (
    <Stack gap="2" p="2">
      <Text color="fg.muted" fontSize="2xs" textWrap="pretty">
        {t('widgets.upscale.sharedPromptDescription')}
      </Text>
      <PositivePromptField
        heightPx={values.positivePromptHeightPx}
        loras={values.loras}
        projectId={projectId}
        selectedModel={values.model ?? undefined}
        showSyntaxHighlighting={showSyntaxHighlighting}
        value={promptDraft.positivePrompt}
        onChange={(positivePrompt) => onPatchPromptDraft({ positivePrompt })}
        onResizeEnd={(positivePromptHeightPx) => onPatchValues({ positivePromptHeightPx })}
        onUsePrompt={handleUsePrompt}
      />
      <NegativePromptField
        heightPx={values.negativePromptHeightPx}
        isEnabled={promptDraft.negativePromptEnabled}
        loras={values.loras}
        projectId={projectId}
        selectedModel={values.model ?? undefined}
        showSyntaxHighlighting={showSyntaxHighlighting}
        value={promptDraft.negativePrompt}
        onChange={(negativePrompt) => onPatchPromptDraft({ negativePrompt })}
        onEnabledChange={(negativePromptEnabled) => onPatchPromptDraft({ negativePromptEnabled })}
        onResizeEnd={(negativePromptHeightPx) => onPatchValues({ negativePromptHeightPx })}
      />
    </Stack>
  );
};

const NumericSliderField = ({
  error,
  formatValue,
  helpText,
  label,
  marks,
  numberMax,
  numberMin,
  onChange,
  sliderMax = numberMax,
  sliderMin = numberMin,
  step,
  value,
}: {
  error?: string;
  formatValue?: (value: number) => string;
  helpText?: string;
  label: string;
  marks?: number[];
  numberMax: number;
  numberMin: number;
  onChange: (value: number) => void;
  sliderMax?: number;
  sliderMin?: number;
  step: number;
  value: number;
}) => (
  <Field error={error} helpText={helpText} label={label}>
    <HStack align="center" gap="3">
      <Slider
        aria-label={[label]}
        flex="1"
        formatValue={(nextValue) => (formatValue ? formatValue(nextValue) : String(nextValue))}
        marks={marks}
        max={sliderMax}
        min={sliderMin}
        step={step}
        value={[Math.min(sliderMax, Math.max(sliderMin, value))]}
        onValueChange={(details) => {
          const nextValue = details.value[0];

          if (nextValue !== undefined) {
            onChange(nextValue);
          }
        }}
      />
      <NumberInput.Root
        max={numberMax}
        min={numberMin}
        size="xs"
        step={step}
        value={String(value)}
        w="20"
        onValueChange={({ valueAsNumber }) => {
          if (Number.isFinite(valueAsNumber)) {
            onChange(valueAsNumber);
          }
        }}
      >
        <NumberInput.Control />
        <NumberInput.Input aria-label={`${label} value`} fontVariantNumeric="tabular-nums" />
      </NumberInput.Root>
    </HStack>
  </Field>
);

const UpscaleImageField = ({
  inputImage,
  onChange,
}: {
  inputImage: UpscaleWidgetValues['inputImage'];
  onChange: (image: UpscaleWidgetValues['inputImage']) => void;
}) => {
  const { t } = useTranslation();
  const { reportError, touchGalleryImages } = useUpscaleUi();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { isOver, setNodeRef } = useDroppable({ data: { kind: DROP_ID }, id: DROP_ID });
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const setGalleryImage = useCallback(
    async (imageName: string) => {
      setErrorMessage(null);
      setIsLoading(true);

      try {
        const [image] = await galleryImages.resolveMany([imageName]);

        if (image) {
          onChange({ height: image.height, image_name: image.imageName, width: image.width });
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setErrorMessage(message);
        reportError(message);
      } finally {
        setIsLoading(false);
      }
    },
    [onChange, reportError]
  );

  useDndMonitor({
    onDragEnd: (event) => {
      const data = event.active.data.current;

      if (!isLoading && event.over?.id === DROP_ID && isGalleryImageDragData(data) && data.images.length === 1) {
        const imageName = data.images[0]?.imageName;

        if (imageName) {
          void setGalleryImage(imageName);
        }
      }
    },
  });

  const uploadFile = useCallback(
    async (file: File) => {
      setErrorMessage(null);

      if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
        setErrorMessage(t('widgets.upscale.unsupportedFile'));
        reportError(t('widgets.upscale.unsupportedFile'));
        return;
      }

      setIsLoading(true);

      try {
        const image = await galleryTransfers.upload(file, 'none');

        onChange({ height: image.height, image_name: image.imageName, width: image.width });
        touchGalleryImages();
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setErrorMessage(message);
        reportError(message);
      } finally {
        setIsLoading(false);
      }
    },
    [onChange, reportError, t, touchGalleryImages]
  );

  const handleFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];

      if (file) {
        void uploadFile(file);
      }
      event.currentTarget.value = '';
    },
    [uploadFile]
  );

  return (
    <Stack gap="2">
      <DropZone
        ref={setNodeRef}
        as="button"
        aria-busy={isLoading}
        aria-label={inputImage ? t('widgets.upscale.replaceImage') : t('widgets.upscale.uploadImage')}
        cursor="pointer"
        isOver={isOver}
        {...(isLoading ? { disabled: true } : undefined)}
        minH="24"
        overflow="hidden"
        position="relative"
        _focusVisible={{
          outlineColor: 'accent.focusRing',
          outlineOffset: '2px',
          outlineStyle: 'solid',
          outlineWidth: '2px',
        }}
        _disabled={{ cursor: 'wait', opacity: 0.7 }}
        _hover={isLoading ? undefined : { bg: 'bg.muted', color: 'fg' }}
        onClick={() => fileInputRef.current?.click()}
      >
        {inputImage ? (
          <HStack align="stretch" gap="3" h="24" p="2">
            <Box bg="blackAlpha.300" boxSize="20" flexShrink="0" overflow="hidden" rounded="sm">
              <Image
                alt={t('widgets.upscale.inputImageAlt')}
                boxSize="full"
                objectFit="contain"
                outline="1px solid rgba(0, 0, 0, 0.1)"
                outlineOffset="-1px"
                rounded="sm"
                src={galleryImageUrls.thumbnail(inputImage.image_name)}
                _dark={{ outlineColor: 'rgba(255, 255, 255, 0.1)' }}
              />
            </Box>
            <Stack align="start" flex="1" gap="1" justify="center" minW="0">
              <Text color="fg" fontSize="xs" fontWeight="semibold" truncate>
                {inputImage.image_name}
              </Text>
              <Text color="fg.muted" fontSize="2xs" fontVariantNumeric="tabular-nums">
                {inputImage.width} × {inputImage.height}
              </Text>
              <HStack color="fg.muted" gap="1">
                {isLoading ? <Spinner size="xs" /> : <UploadIcon aria-hidden="true" size="12" />}
                <Text fontSize="2xs">
                  {isLoading ? t('widgets.upscale.uploadingImage') : t('widgets.upscale.replaceOrDrop')}
                </Text>
              </HStack>
            </Stack>
          </HStack>
        ) : (
          <Stack align="center" color="fg.muted" gap="2" justify="center" minH="24" px="4">
            {isLoading ? <Spinner size="sm" /> : <ImagePlusIcon aria-hidden="true" size="20" />}
            <Text fontSize="xs" textAlign="center">
              {isLoading ? t('widgets.upscale.uploadingImage') : t('widgets.upscale.uploadOrDrop')}
            </Text>
          </Stack>
        )}
      </DropZone>
      <HStack justify="end">
        {inputImage ? (
          <Button disabled={isLoading} size="xs" variant="ghost" onClick={() => onChange(null)}>
            <XIcon aria-hidden="true" size="12" />
            {t('widgets.upscale.removeImage')}
          </Button>
        ) : null}
      </HStack>
      {errorMessage ? (
        <Text aria-live="polite" color="fg.error" fontSize="2xs" role="alert" textWrap="pretty">
          {errorMessage}
        </Text>
      ) : null}
      <Input
        ref={fileInputRef}
        accept="image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp"
        aria-hidden="true"
        display="none"
        tabIndex={-1}
        type="file"
        onChange={handleFileChange}
      />
    </Stack>
  );
};

export const UpscaleWidgetView = () => {
  const { t } = useTranslation();
  const selection = useUpscaleUi();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const normalized = normalizeUpscaleWidgetValues(selection.rawValues) ?? createDefaultUpscaleWidgetValues();
  const values = modelsStatus === 'loaded' ? syncUpscaleWidgetValuesWithModels(normalized, models) : normalized;
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
  const errors = {
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
  };
  const patch = useCallback((next: Partial<UpscaleWidgetValues>) => selection.patchValues(next), [selection]);
  const patchPromptDraft = useCallback(
    (next: ProjectPromptDraftPatch) => selection.patchPromptDraft(next),
    [selection]
  );
  const replace = useCallback((next: UpscaleWidgetValues) => selection.patchValues({ ...next }), [selection]);

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const selectMainModel = (model: ModelConfig | null) => {
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

    replace(nextValues);

    if (notices.length > 0) {
      toaster.create({
        description: notices.join(' '),
        title: t('widgets.upscale.settingsAdjusted'),
        type: 'info',
      });
    }
  };

  const addLora = (model: ModelConfig | null) => {
    if (!values.model || !isLoraModelConfig(model) || !isLoraCompatibleWithModel(model, values.model)) {
      return;
    }

    patch({ loras: [...values.loras, { isEnabled: true, model, weight: getDefaultLoraWeight(model) }] });
  };

  const updateLora = (key: string, update: Partial<GenerateLora>) =>
    patch({ loras: values.loras.map((lora) => (lora.model.key === key ? { ...lora, ...update } : lora)) });
  const selectedLoraKeys = new Set(values.loras.map((lora) => lora.model.key));

  const sharedBadge = (
    <Badge fontFamily="mono" size="xs">
      {t('widgets.upscale.shared')}
    </Badge>
  );

  return (
    <Stack gap="1" minW="0" p="1">
      <UpscaleModelReconciler
        key={`${selection.projectId}:${modelsStatus}:${modelsFingerprint}`}
        rawValues={selection.rawValues}
        values={values}
      />

      <GenerationSettingsSection label={t('widgets.upscale.sourceAndTreatment')} defaultOpen>
        <Stack gap="3" p="2">
          <UpscaleImageField inputImage={values.inputImage} onChange={(inputImage) => patch({ inputImage })} />
          <UpscaleOutputPreflight values={values} />
          <Field
            error={values.upscaleModel ? undefined : t('widgets.upscale.spandrelModelRequired')}
            helpText={values.upscaleModel ? t('widgets.upscale.spandrelModelHelp') : undefined}
            label={t('widgets.upscale.spandrelModel')}
          >
            <ModelSelect
              invalid={!values.upscaleModel}
              modelTypes={['spandrel_image_to_image']}
              placeholder={t('widgets.upscale.selectSpandrelModel')}
              size="xs"
              value={values.upscaleModel?.key ?? null}
              onChange={(model) => patch({ upscaleModel: isSpandrelModelConfig(model) ? model : null })}
            />
          </Field>
          <NumericSliderField
            error={errors.scale}
            formatValue={(scale) => `${scale}×`}
            helpText={t('widgets.upscale.scaleHelp')}
            label={t('widgets.upscale.scale')}
            marks={[1, 2, 4, 8, 16]}
            numberMax={UPSCALE_SCALE_MAX}
            numberMin={UPSCALE_SCALE_MIN}
            step={0.5}
            value={values.scale}
            onChange={(scale) => patch({ scale })}
          />
          <ButtonGroup attached={false} size="xs" variant="outline">
            <SimpleGrid columns={{ base: 2, md: 4 }} gap="1" w="full">
              {Object.entries(UPSCALE_PRESETS).map(([id, preset]) => {
                const active = values.creativity === preset.creativity && values.structure === preset.structure;
                const tooltipContent = `${t(`widgets.upscale.presetDescriptions.${id}`)} ${t(
                  'widgets.upscale.presetValues',
                  { creativity: preset.creativity, structure: preset.structure }
                )}`;

                return (
                  <Tooltip key={id} content={tooltipContent}>
                    <Button
                      aria-pressed={active}
                      colorPalette={active ? 'accent' : 'bg'}
                      size="xs"
                      variant={active ? 'solid' : 'outline'}
                      onClick={() => patch(preset)}
                    >
                      {t(`widgets.upscale.presets.${id}`)}
                    </Button>
                  </Tooltip>
                );
              })}
            </SimpleGrid>
          </ButtonGroup>
          <NumericSliderField
            error={errors.creativity}
            helpText={t('widgets.upscale.creativityHelp')}
            label={t('widgets.upscale.creativity')}
            marks={[UPSCALE_CREATIVITY_MIN, 0, UPSCALE_CREATIVITY_MAX]}
            numberMax={UPSCALE_CREATIVITY_MAX}
            numberMin={UPSCALE_CREATIVITY_MIN}
            step={1}
            value={values.creativity}
            onChange={(creativity) => patch({ creativity })}
          />
          <NumericSliderField
            error={errors.structure}
            helpText={t('widgets.upscale.structureHelp')}
            label={t('widgets.upscale.structure')}
            marks={[UPSCALE_STRUCTURE_MIN, 0, UPSCALE_STRUCTURE_MAX]}
            numberMax={UPSCALE_STRUCTURE_MAX}
            numberMin={UPSCALE_STRUCTURE_MIN}
            step={1}
            value={values.structure}
            onChange={(structure) => patch({ structure })}
          />
        </Stack>
      </GenerationSettingsSection>

      <GenerationSettingsSection badges={sharedBadge} label={t('widgets.upscale.detailGuidance')}>
        <UpscalePromptFields
          promptDraft={selection.promptDraft}
          projectId={selection.projectId}
          showSyntaxHighlighting={selection.showPromptSyntaxHighlighting}
          values={values}
          onPatchPromptDraft={patchPromptDraft}
          onPatchValues={patch}
        />
      </GenerationSettingsSection>

      <GenerationSettingsSection label={t('widgets.upscale.generation')}>
        <Stack gap="3" p="2">
          <Field
            error={values.model ? undefined : t('widgets.upscale.mainModelRequired')}
            label={t('widgets.upscale.mainModel')}
          >
            <ModelSelect
              filter={isSelectableMainModel}
              invalid={!values.model}
              modelTypes={['main']}
              placeholder={t('widgets.upscale.selectMainModel')}
              size="xs"
              value={values.model?.key ?? null}
              onChange={selectMainModel}
            />
          </Field>
          <SimpleGrid columns={{ base: 2, md: 3 }} gap="2">
            <Field error={errors.steps} label={t('widgets.upscale.steps')}>
              <NumberInput.Root
                max={1000}
                min={1}
                size="xs"
                value={String(values.steps)}
                onValueChange={({ valueAsNumber }) => Number.isFinite(valueAsNumber) && patch({ steps: valueAsNumber })}
              >
                <NumberInput.Control />
                <NumberInput.Input fontVariantNumeric="tabular-nums" />
              </NumberInput.Root>
            </Field>
            <Field error={errors.cfgScale} label={t('widgets.upscale.cfgScale')}>
              <NumberInput.Root
                max={100}
                min={0}
                size="xs"
                step={0.5}
                value={String(values.cfgScale)}
                onValueChange={({ valueAsNumber }) =>
                  Number.isFinite(valueAsNumber) && patch({ cfgScale: valueAsNumber })
                }
              >
                <NumberInput.Control />
                <NumberInput.Input fontVariantNumeric="tabular-nums" />
              </NumberInput.Root>
            </Field>
            <Field label={t('widgets.upscale.batchCount')}>
              <NumberInput.Root
                min={1}
                size="xs"
                value={String(values.batchCount)}
                onValueChange={({ valueAsNumber }) =>
                  Number.isFinite(valueAsNumber) && patch({ batchCount: valueAsNumber })
                }
              >
                <NumberInput.Control />
                <NumberInput.Input fontVariantNumeric="tabular-nums" />
              </NumberInput.Root>
            </Field>
          </SimpleGrid>
          <Field label={t('widgets.upscale.scheduler')}>
            <Combobox
              aria-label={t('widgets.upscale.scheduler')}
              options={SCHEDULER_OPTIONS}
              size="xs"
              value={values.scheduler}
              onValueChange={(scheduler) => patch({ scheduler })}
            />
          </Field>
          <Field error={values.shouldRandomizeSeed ? undefined : errors.seed} label={t('widgets.upscale.seed')}>
            <HStack gap="2">
              <NumberInput.Root
                disabled={values.shouldRandomizeSeed}
                max={SEED_MAX}
                min={0}
                size="xs"
                value={String(values.seed)}
                w="full"
                onValueChange={({ valueAsNumber }) => Number.isFinite(valueAsNumber) && patch({ seed: valueAsNumber })}
              >
                <NumberInput.Input fontVariantNumeric="tabular-nums" />
              </NumberInput.Root>
              <Tooltip content={t('widgets.upscale.shuffleSeed')}>
                <IconButton
                  aria-label={t('widgets.upscale.shuffleSeed')}
                  disabled={values.shouldRandomizeSeed}
                  size="xs"
                  variant="outline"
                  onClick={() => patch({ seed: Math.floor(Math.random() * SEED_MAX) })}
                >
                  <DicesIcon />
                </IconButton>
              </Tooltip>
              <Switch.Root
                checked={values.shouldRandomizeSeed}
                size="sm"
                onCheckedChange={(event) => patch({ shouldRandomizeSeed: event.checked })}
              >
                <Switch.HiddenInput />
                <Switch.Control _checked={{ bg: 'accent.solid' }}>
                  <Switch.Thumb />
                </Switch.Control>
                <Switch.Label fontSize="xs">{t('widgets.upscale.random')}</Switch.Label>
              </Switch.Root>
            </HStack>
          </Field>
          <Field label={t('widgets.upscale.addLora')}>
            <ModelSelect
              excludeKeys={selectedLoraKeys}
              filter={(model) =>
                Boolean(values.model && isLoraModelConfig(model) && isLoraCompatibleWithModel(model, values.model))
              }
              modelTypes={['lora']}
              placeholder={t('widgets.upscale.selectLora')}
              size="xs"
              value={null}
              onChange={addLora}
            />
          </Field>
          {values.loras.map((lora) => (
            <HStack key={lora.model.key} bg="bg.subtle" gap="2" p="2" rounded="md">
              <Switch.Root
                aria-label={lora.model.name}
                checked={lora.isEnabled}
                size="sm"
                onCheckedChange={(event) => updateLora(lora.model.key, { isEnabled: event.checked })}
              >
                <Switch.HiddenInput />
                <Switch.Control _checked={{ bg: 'accent.solid' }}>
                  <Switch.Thumb />
                </Switch.Control>
              </Switch.Root>
              <Text flex="1" fontSize="xs" minW="0" truncate>
                {lora.model.name}
              </Text>
              <NumberInput.Root
                max={10}
                min={-10}
                size="xs"
                step={0.05}
                value={String(lora.weight)}
                w="20"
                onValueChange={({ valueAsNumber }) =>
                  Number.isFinite(valueAsNumber) && updateLora(lora.model.key, { weight: valueAsNumber })
                }
              >
                <NumberInput.Input aria-label={t('widgets.upscale.loraWeight', { name: lora.model.name })} />
              </NumberInput.Root>
              <IconButton
                aria-label={t('widgets.upscale.removeLora', { name: lora.model.name })}
                size="xs"
                variant="ghost"
                onClick={() =>
                  patch({ loras: values.loras.filter((candidate) => candidate.model.key !== lora.model.key) })
                }
              >
                <Trash2Icon />
              </IconButton>
            </HStack>
          ))}
        </Stack>
      </GenerationSettingsSection>

      <GenerationSettingsSection label={t('widgets.upscale.advanced')}>
        <Stack gap="3" p="2">
          <Field
            error={values.tileControlnetModel ? undefined : t('widgets.upscale.tileControlNetRequired')}
            helpText={values.tileControlnetModel ? t('widgets.upscale.tileControlNetHelp') : undefined}
            label={t('widgets.upscale.tileControlNet')}
          >
            <ModelSelect
              filter={(model) => isTileControlNetCandidate(model, values.model)}
              invalid={!values.tileControlnetModel}
              modelTypes={['controlnet']}
              placeholder={t('widgets.upscale.selectTileControlNet')}
              size="xs"
              value={values.tileControlnetModel?.key ?? null}
              onChange={(model) =>
                patch({ tileControlnetModel: isTileControlNetCandidate(model, values.model) ? model : null })
              }
            />
          </Field>
          <NumericSliderField
            error={errors.tileSize}
            helpText={t('widgets.upscale.tileSizeHelp')}
            label={t('widgets.upscale.tileSize')}
            marks={[UPSCALE_TILE_SIZE_MIN, 1024, UPSCALE_TILE_SIZE_MAX]}
            numberMax={UPSCALE_TILE_SIZE_MAX}
            numberMin={UPSCALE_TILE_SIZE_MIN}
            step={64}
            value={values.tileSize}
            onChange={(tileSize) => patch({ tileSize })}
          />
          <NumericSliderField
            error={errors.tileOverlap}
            helpText={t('widgets.upscale.tileOverlapHelp')}
            label={t('widgets.upscale.tileOverlap')}
            marks={[UPSCALE_TILE_OVERLAP_MIN, 128, 256, UPSCALE_TILE_OVERLAP_MAX]}
            numberMax={UPSCALE_TILE_OVERLAP_MAX}
            numberMin={UPSCALE_TILE_OVERLAP_MIN}
            step={8}
            value={values.tileOverlap}
            onChange={(tileOverlap) => patch({ tileOverlap })}
          />
          <SimpleGrid columns={{ base: 1, md: 2 }} gap="2">
            <Field label={t('widgets.upscale.vae')} helpText={values.vae ? undefined : t('widgets.upscale.bundledVae')}>
              <ModelSelect
                filter={(model) => Boolean(values.model && model.base === values.model.base)}
                isClearable
                modelTypes={['vae']}
                placeholder={t('widgets.upscale.bundledVae')}
                size="xs"
                value={values.vae?.key ?? null}
                onChange={(model) => patch({ vae: isVaeModelConfig(model) ? model : null })}
              />
            </Field>
            <Field label={t('widgets.upscale.vaePrecision')}>
              <Select
                aria-label={t('widgets.upscale.vaePrecision')}
                collection={VAE_PRECISION_COLLECTION}
                size="xs"
                value={[values.vaePrecision]}
                onValueChange={({ value }) => {
                  const vaePrecision = value[0];

                  if (vaePrecision === 'fp16' || vaePrecision === 'fp32') {
                    patch({ vaePrecision });
                  }
                }}
              />
            </Field>
          </SimpleGrid>
          {values.model?.base === 'sd-1' ? (
            <Field label={t('widgets.upscale.clipSkip')}>
              <NumberInput.Root
                max={12}
                min={0}
                size="xs"
                value={String(values.clipSkip)}
                onValueChange={({ valueAsNumber }) =>
                  Number.isFinite(valueAsNumber) && patch({ clipSkip: valueAsNumber })
                }
              >
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

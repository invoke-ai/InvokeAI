import type { DragEndEvent } from '@dnd-kit/core';
import type { GenerateLora, MainModelConfig, PromptHistoryItem } from '@features/generation/contracts';
import type { ProjectPromptDraft, ProjectPromptDraftPatch } from '@features/generation/settings';
import type { ModelConfig, ModelTaxonomyType } from '@features/models';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';
import type { FeatureHintId } from '@platform/ui/hints';
import type { ChangeEvent } from 'react';

import {
  Badge,
  Box,
  createListCollection,
  DataList,
  HStack,
  Image,
  Input,
  NumberInput,
  SegmentGroup,
  SimpleGrid,
  Spinner,
  Stack,
  Switch,
  Text,
} from '@chakra-ui/react';
import { useDndMonitor } from '@dnd-kit/core';
import { galleryImages, galleryTransfers } from '@features/gallery';
import { galleryImageUrls, isGalleryImageDragData, useGalleryImageDroppable } from '@features/gallery/utility';
import { GenerationSettingsSection, NegativePromptField, PositivePromptField } from '@features/generation/components';
import {
  areProjectPromptDraftsEqual,
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
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { Button, Combobox, DropZone, Field, IconButton, Select, Slider, Tooltip } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { toaster } from '@platform/ui/toaster';
import { DicesIcon, ImagePlusIcon, Trash2Icon, UploadIcon, XIcon } from 'lucide-react';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

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
const DROP_ZONE_FOCUS_PROPS = {
  outlineColor: 'accent.focusRing',
  outlineOffset: '2px',
  outlineStyle: 'solid',
  outlineWidth: '2px',
};
const DROP_ZONE_DISABLED_PROPS = { cursor: 'wait', opacity: 0.7 };
// `DropZone` types its props as `BoxProps`, which has no `disabled`; spreading a
// hoisted object keeps the button attribute without a per-render object.
const DROP_ZONE_BUSY_PROPS = { disabled: true };
const DROP_ZONE_HOVER_PROPS = { bg: 'bg.muted', color: 'fg' };
const UPLOAD_ACCEPT_TYPES = ['image/png', 'image/jpeg', 'image/webp'];
const PRESET_ENTRIES = Object.entries(UPSCALE_PRESETS);

const isSelectableMainModel = (model: ModelConfig): boolean => isSupportedUpscaleMainModel(model);

const formatScale = (scale: number): string => `${scale}×`;

const valuesAreEqual = (left: UpscaleWidgetValues, right: UpscaleWidgetValues): boolean =>
  JSON.stringify(left) === JSON.stringify(right);

const getRangeError = (label: string, value: number, min: number, max: number): string | undefined =>
  Number.isFinite(value) && value >= min && value <= max ? undefined : `${label} must be between ${min} and ${max}.`;

/**
 * `values` is re-derived from the raw widget state on every patch, so its
 * nested members arrive with fresh identities even when nothing about them
 * changed. Sections whose props are those members compare by content instead,
 * which is what keeps a scale edit from re-rendering the prompt editors.
 */
const areLorasEquivalent = (left: readonly GenerateLora[], right: readonly GenerateLora[]): boolean =>
  left.length === right.length &&
  left.every((lora, index) => {
    const other = right[index];

    return (
      other !== undefined &&
      lora.model.key === other.model.key &&
      lora.isEnabled === other.isEnabled &&
      lora.weight === other.weight
    );
  });

const areInputImagesEquivalent = (
  left: UpscaleWidgetValues['inputImage'],
  right: UpscaleWidgetValues['inputImage']
): boolean =>
  left === right ||
  (left !== null &&
    right !== null &&
    left.image_name === right.image_name &&
    left.width === right.width &&
    left.height === right.height);

const UpscaleOutputPreflight = memo(function UpscaleOutputPreflight({
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
});

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

const UpscalePromptFields = memo(
  function UpscalePromptFields({
    loras,
    model,
    negativePromptHeightPx,
    onPatchPromptDraft,
    onPatchValues,
    positivePromptHeightPx,
    promptDraft,
    projectId,
    showSyntaxHighlighting,
  }: {
    loras: GenerateLora[];
    model: MainModelConfig | null;
    negativePromptHeightPx: number;
    onPatchPromptDraft: (patch: ProjectPromptDraftPatch) => void;
    onPatchValues: (patch: Partial<UpscaleWidgetValues>) => void;
    positivePromptHeightPx: number;
    promptDraft: ProjectPromptDraft;
    projectId: string;
    showSyntaxHighlighting: boolean;
  }) {
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
    const handlePositiveChange = useCallback(
      (positivePrompt: string) => onPatchPromptDraft({ positivePrompt }),
      [onPatchPromptDraft]
    );
    const handleNegativeChange = useCallback(
      (negativePrompt: string) => onPatchPromptDraft({ negativePrompt }),
      [onPatchPromptDraft]
    );
    const handleNegativeEnabledChange = useCallback(
      (negativePromptEnabled: boolean) => onPatchPromptDraft({ negativePromptEnabled }),
      [onPatchPromptDraft]
    );
    const handlePositiveResizeEnd = useCallback(
      (positivePromptHeight: number) => onPatchValues({ positivePromptHeightPx: positivePromptHeight }),
      [onPatchValues]
    );
    const handleNegativeResizeEnd = useCallback(
      (negativePromptHeight: number) => onPatchValues({ negativePromptHeightPx: negativePromptHeight }),
      [onPatchValues]
    );

    return (
      <Stack gap="2" p="2">
        <Text color="fg.muted" fontSize="2xs" textWrap="pretty">
          {t('widgets.upscale.sharedPromptDescription')}
        </Text>
        <PositivePromptField
          heightPx={positivePromptHeightPx}
          loras={loras}
          projectId={projectId}
          selectedModel={model ?? undefined}
          showSyntaxHighlighting={showSyntaxHighlighting}
          value={promptDraft.positivePrompt}
          onChange={handlePositiveChange}
          onResizeEnd={handlePositiveResizeEnd}
          onUsePrompt={handleUsePrompt}
        />
        <NegativePromptField
          heightPx={negativePromptHeightPx}
          isEnabled={promptDraft.negativePromptEnabled}
          loras={loras}
          projectId={projectId}
          selectedModel={model ?? undefined}
          showSyntaxHighlighting={showSyntaxHighlighting}
          value={promptDraft.negativePrompt}
          onChange={handleNegativeChange}
          onEnabledChange={handleNegativeEnabledChange}
          onResizeEnd={handleNegativeResizeEnd}
        />
      </Stack>
    );
  },
  (previous, next) =>
    previous.negativePromptHeightPx === next.negativePromptHeightPx &&
    previous.onPatchPromptDraft === next.onPatchPromptDraft &&
    previous.onPatchValues === next.onPatchValues &&
    previous.positivePromptHeightPx === next.positivePromptHeightPx &&
    previous.projectId === next.projectId &&
    areProjectPromptDraftsEqual(previous.promptDraft, next.promptDraft) &&
    previous.showSyntaxHighlighting === next.showSyntaxHighlighting &&
    previous.model?.key === next.model?.key &&
    areLorasEquivalent(previous.loras, next.loras)
);

const NumericSliderField = memo(function NumericSliderField({
  error,
  formatValue,
  helpText,
  hint,
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
  hint?: FeatureHintId;
  label: string;
  marks?: number[];
  numberMax: number;
  numberMin: number;
  onChange: (value: number) => void;
  sliderMax?: number;
  sliderMin?: number;
  step: number;
  value: number;
}) {
  const ariaLabel = useMemo(() => [label], [label]);
  const sliderValue = useMemo(() => [Math.min(sliderMax, Math.max(sliderMin, value))], [sliderMax, sliderMin, value]);
  const handleFormatValue = useCallback(
    (nextValue: number) => (formatValue ? formatValue(nextValue) : String(nextValue)),
    [formatValue]
  );
  const handleSliderChange = useCallback(
    (details: { value: number[] }) => {
      const nextValue = details.value[0];

      if (nextValue !== undefined) {
        onChange(nextValue);
      }
    },
    [onChange]
  );
  const handleNumberChange = useCallback(
    ({ valueAsNumber }: NumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        onChange(valueAsNumber);
      }
    },
    [onChange]
  );

  return (
    <Field error={error} helpText={helpText} hint={hint} label={label}>
      <HStack align="center" gap="3">
        <Slider
          aria-label={ariaLabel}
          flex="1"
          formatValue={handleFormatValue}
          marks={marks}
          max={sliderMax}
          min={sliderMin}
          size="sm"
          step={step}
          value={sliderValue}
          onValueChange={handleSliderChange}
        />
        <NumberInput.Root
          max={numberMax}
          min={numberMin}
          size="xs"
          step={step}
          value={String(value)}
          w="20"
          onValueChange={handleNumberChange}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={`${label} value`} fontVariantNumeric="tabular-nums" />
        </NumberInput.Root>
      </HStack>
    </Field>
  );
});

const UpscaleImageField = memo(
  function UpscaleImageField({
    inputImage,
    onChange,
  }: {
    inputImage: UpscaleWidgetValues['inputImage'];
    onChange: (image: UpscaleWidgetValues['inputImage']) => void;
  }) {
    const { t } = useTranslation();
    const { reportError, touchGalleryImages } = useUpscaleUiActions();
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const { isOver, setNodeRef } = useGalleryImageDroppable({
      data: { kind: DROP_ID },
      disabled: isLoading,
      id: DROP_ID,
    });

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

    const handleDragEnd = useCallback(
      (event: DragEndEvent) => {
        const data = event.active.data.current;

        if (!isLoading && event.over?.id === DROP_ID && isGalleryImageDragData(data) && data.items.length === 1) {
          const imageName = data.items[0]?.name;

          if (imageName) {
            void setGalleryImage(imageName);
          }
        }
      },
      [isLoading, setGalleryImage]
    );

    useDndMonitor({ onDragEnd: handleDragEnd });

    const uploadFile = useCallback(
      async (file: File) => {
        setErrorMessage(null);

        if (!UPLOAD_ACCEPT_TYPES.includes(file.type)) {
          setErrorMessage(t('widgets.upscale.unsupportedFile'));
          reportError(t('widgets.upscale.unsupportedFile'));
          return;
        }

        const owner = captureAccountScope();
        setIsLoading(true);

        try {
          const image = await galleryTransfers.upload(file, 'none', { signal: owner.signal });

          assertAccountScopeCurrent(owner);
          onChange({ height: image.height, image_name: image.imageName, width: image.width });
          touchGalleryImages();
        } catch (error) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

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
    const handlePickFile = useCallback(() => fileInputRef.current?.click(), []);
    const handleClear = useCallback(() => onChange(null), [onChange]);

    return (
      <Stack gap="2">
        <DropZone
          ref={setNodeRef}
          as="button"
          aria-busy={isLoading}
          aria-label={inputImage ? t('widgets.upscale.replaceImage') : t('widgets.upscale.uploadImage')}
          cursor="pointer"
          isOver={isOver}
          {...(isLoading ? DROP_ZONE_BUSY_PROPS : undefined)}
          minH="24"
          overflow="hidden"
          position="relative"
          _focusVisible={DROP_ZONE_FOCUS_PROPS}
          _disabled={DROP_ZONE_DISABLED_PROPS}
          _hover={isLoading ? undefined : DROP_ZONE_HOVER_PROPS}
          onClick={handlePickFile}
        >
          {inputImage ? (
            <HStack align="stretch" gap="3" h="24" p="2">
              <Box bg="blackAlpha.300" boxSize="20" flexShrink="0" overflow="hidden" rounded="sm">
                <Image
                  alt={t('widgets.upscale.inputImageAlt')}
                  boxSize="full"
                  objectFit="contain"
                  outline="1px solid"
                  outlineColor="border.image"
                  outlineOffset="-1px"
                  rounded="sm"
                  src={galleryImageUrls.thumbnail(inputImage.image_name)}
                />
              </Box>
              <Stack align="start" flex="1" gap="1" justify="center" minW="0">
                <MiddleTruncate color="fg" fontSize="xs" fontWeight="semibold" text={inputImage.image_name} />
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
            <Button disabled={isLoading} size="xs" variant="ghost" onClick={handleClear}>
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
  },
  (previous, next) =>
    previous.onChange === next.onChange && areInputImagesEquivalent(previous.inputImage, next.inputImage)
);

/**
 * One LoRA row. Split out so editing a single weight re-renders that row rather
 * than the whole list; the handlers bind the key here instead of at the call
 * site, where they would be new closures per row per render.
 */
const UpscaleLoraRow = memo(function UpscaleLoraRow({
  lora,
  onRemove,
  onUpdate,
}: {
  lora: GenerateLora;
  onRemove: (key: string) => void;
  onUpdate: (key: string, update: Partial<GenerateLora>) => void;
}) {
  const { t } = useTranslation();
  const modelKey = lora.model.key;
  const handleToggle = useCallback(
    (details: { checked: boolean }) => onUpdate(modelKey, { isEnabled: details.checked }),
    [modelKey, onUpdate]
  );
  const handleWeightChange = useCallback(
    ({ valueAsNumber }: NumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        onUpdate(modelKey, { weight: valueAsNumber });
      }
    },
    [modelKey, onUpdate]
  );
  const handleRemove = useCallback(() => onRemove(modelKey), [modelKey, onRemove]);

  return (
    <HStack bg="bg.subtle" gap="2" p="2" rounded="md">
      <Switch.Root aria-label={lora.model.name} checked={lora.isEnabled} size="sm" onCheckedChange={handleToggle}>
        <Switch.HiddenInput />
        <Switch.Control _checked={SWITCH_CHECKED_PROPS}>
          <Switch.Thumb />
        </Switch.Control>
      </Switch.Root>
      <MiddleTruncate flex="1" fontSize="xs" minW="0" text={lora.model.name} />
      <NumberInput.Root
        max={10}
        min={-10}
        size="xs"
        step={0.05}
        value={String(lora.weight)}
        w="20"
        onValueChange={handleWeightChange}
      >
        <NumberInput.Input aria-label={t('widgets.upscale.loraWeight', { name: lora.model.name })} />
      </NumberInput.Root>
      <IconButton
        aria-label={t('widgets.upscale.removeLora', { name: lora.model.name })}
        size="xs"
        variant="ghost"
        onClick={handleRemove}
      >
        <Trash2Icon />
      </IconButton>
    </HStack>
  );
});

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
          <NumericSliderField
            error={errors.scale}
            formatValue={formatScale}
            helpText={t('widgets.upscale.scaleHelp')}
            hint="upscaleScale"
            label={t('widgets.upscale.scale')}
            marks={SCALE_MARKS}
            numberMax={UPSCALE_SCALE_MAX}
            numberMin={UPSCALE_SCALE_MIN}
            step={0.5}
            value={values.scale}
            onChange={set.scale}
          />
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
          <NumericSliderField
            error={errors.creativity}
            helpText={t('widgets.upscale.creativityHelp')}
            hint="creativity"
            label={t('widgets.upscale.creativity')}
            marks={CREATIVITY_MARKS}
            numberMax={UPSCALE_CREATIVITY_MAX}
            numberMin={UPSCALE_CREATIVITY_MIN}
            step={1}
            value={values.creativity}
            onChange={set.creativity}
          />
          <NumericSliderField
            error={errors.structure}
            helpText={t('widgets.upscale.structureHelp')}
            hint="structure"
            label={t('widgets.upscale.structure')}
            marks={STRUCTURE_MARKS}
            numberMax={UPSCALE_STRUCTURE_MAX}
            numberMin={UPSCALE_STRUCTURE_MIN}
            step={1}
            value={values.structure}
            onChange={set.structure}
          />
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
          <NumericSliderField
            error={errors.tileSize}
            helpText={t('widgets.upscale.tileSizeHelp')}
            hint="tileSize"
            label={t('widgets.upscale.tileSize')}
            marks={TILE_SIZE_MARKS}
            numberMax={UPSCALE_TILE_SIZE_MAX}
            numberMin={UPSCALE_TILE_SIZE_MIN}
            step={64}
            value={values.tileSize}
            onChange={set.tileSize}
          />
          <NumericSliderField
            error={errors.tileOverlap}
            helpText={t('widgets.upscale.tileOverlapHelp')}
            hint="tileOverlap"
            label={t('widgets.upscale.tileOverlap')}
            marks={TILE_OVERLAP_MARKS}
            numberMax={UPSCALE_TILE_OVERLAP_MAX}
            numberMin={UPSCALE_TILE_OVERLAP_MIN}
            step={8}
            value={values.tileOverlap}
            onChange={set.tileOverlap}
          />
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

import type { SelectValueChangeDetails, SliderValueChangeDetails } from '@chakra-ui/react';
import type { ModelConfig } from '@features/models';
import type {
  CanvasMaskContract,
  CanvasMaskFillContract,
  CanvasRegionalGuidanceLayerContract,
  RegionalGuidanceIPAdapterMethod,
  RegionalGuidanceReferenceImage,
  RegionalGuidanceReferenceImageAsset,
} from '@workbench/canvas-engine/api';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';
import type { ChangeEvent, CSSProperties, FocusEvent, KeyboardEvent } from 'react';

import { Box, createListCollection, HStack, IconButton, Input, Stack, Switch, Text } from '@chakra-ui/react';
import { useDndMonitor, useDroppable } from '@dnd-kit/core';
import { galleryImages, galleryTransfers } from '@features/gallery';
import { isGalleryImageDragData } from '@features/gallery/utility';
import { FluxReduxControls, PROMPT_ATTENTION_TARGET_PROPS, PromptTextarea } from '@features/generation/components';
import { useModelsSelector } from '@features/models';
import { Button, ColorPicker, Field, Select, Slider } from '@platform/ui';
import { useCanvasProjectMutationDispatch } from '@workbench/useCanvasProjectMutationDispatch';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { ImageIcon, PlusIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { applyStructural, applyStructuralPreview, createRegionalReferenceImage } from './layerOps';
import { useSelectedModelBase } from './useSelectedModelBase';

/** The regional-guidance fields patchable via `updateCanvasLayerConfig`. */
interface RegionalConfigPatch {
  mask?: Partial<CanvasMaskContract>;
  positivePrompt?: string | null;
  negativePrompt?: string | null;
  autoNegative?: boolean;
  referenceImages?: RegionalGuidanceReferenceImage[];
}

/** The six mask fill styles, matching `CanvasMaskFillContract['style']` / legacy `zFillStyle`. */
const MASK_FILL_STYLES: readonly CanvasMaskFillContract['style'][] = [
  'solid',
  'grid',
  'crosshatch',
  'diagonal',
  'horizontal',
  'vertical',
];

const IP_ADAPTER_METHODS: readonly RegionalGuidanceIPAdapterMethod[] = [
  'full',
  'style',
  'composition',
  'style_strong',
  'style_precise',
];

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: false } as const;

const COVER_IMG_STYLE: CSSProperties = { height: '100%', objectFit: 'cover', width: '100%' };

const formatWeight = (value: number): string => value.toFixed(2);

const REGIONAL_PROMPT_HEIGHT_PX = 72;

/** DnD droppable id for a specific region's reference-image slot (gallery-image drop target). */
const referenceImageDropId = (layerId: string, refId: string): string => `regional-ref-image:${layerId}:${refId}`;

/**
 * A fresh regional reference image for the currently selected model's base. Shared
 * with the header add-layer menu via `layerOps.createRegionalReferenceImage`.
 */
const createReferenceImage = (base: string | null): RegionalGuidanceReferenceImage =>
  createRegionalReferenceImage(base);

interface RegionalGuidanceSettingsProps {
  engine: CanvasStructuralEngine | null;
  layer: CanvasRegionalGuidanceLayerContract;
}

/**
 * Per-layer settings for a selected regional-guidance region (plan §1.3): a
 * positive + negative prompt, an Auto-Negative toggle, the mask fill colour/style
 * + invert, and a per-region reference-images section (add/remove/enable + model
 * + weight, mirroring the generate widget's IP-Adapter fields). Prompt/toggle/fill
 * edits go through the canvas undo stack (`applyStructural` →
 * `updateCanvasLayerConfig`); invert is an engine pixel op.
 */
export const RegionalGuidanceSettings = ({ engine, layer }: RegionalGuidanceSettingsProps) => {
  const { t } = useTranslation();
  const dispatch = useCanvasProjectMutationDispatch();
  const { gallery, notifications } = useWorkbenchCommands();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const base = useSelectedModelBase();
  const showSyntaxHighlighting = useActiveProjectSelector((project) => project.settings.showPromptSyntaxHighlighting);
  const fillBeforeRef = useRef<CanvasMaskFillContract | null>(null);
  const [positivePrompt, setPositivePrompt] = useState(layer.positivePrompt ?? '');
  const [negativePrompt, setNegativePrompt] = useState(layer.negativePrompt ?? '');

  const fill = layer.mask.fill;
  const isFlux = base === 'flux';

  const commitConfig = useCallback(
    (label: string, next: RegionalConfigPatch, before: RegionalConfigPatch) => {
      applyStructural(
        engine,
        dispatch,
        label,
        { config: { layerType: 'regional_guidance', ...next }, id: layer.id, type: 'updateCanvasLayerConfig' },
        { config: { layerType: 'regional_guidance', ...before }, id: layer.id, type: 'updateCanvasLayerConfig' }
      );
    },
    [dispatch, engine, layer.id]
  );

  const handlePositiveBlur = useCallback(
    (event: FocusEvent<HTMLTextAreaElement>) => {
      const value = event.target.value;
      const next = value.length > 0 ? value : null;
      if (next !== layer.positivePrompt) {
        commitConfig(
          t('widgets.layers.regionalGuidance.positivePrompt'),
          { positivePrompt: next },
          {
            positivePrompt: layer.positivePrompt,
          }
        );
      }
    },
    [commitConfig, layer.positivePrompt, t]
  );

  const handlePositiveChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => setPositivePrompt(event.currentTarget.value),
    []
  );

  const handleNegativeBlur = useCallback(
    (event: FocusEvent<HTMLTextAreaElement>) => {
      const value = event.target.value;
      const next = value.length > 0 ? value : null;
      if (next !== layer.negativePrompt) {
        commitConfig(
          t('widgets.layers.regionalGuidance.negativePrompt'),
          { negativePrompt: next },
          {
            negativePrompt: layer.negativePrompt,
          }
        );
      }
    },
    [commitConfig, layer.negativePrompt, t]
  );

  const handleNegativeChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => setNegativePrompt(event.currentTarget.value),
    []
  );

  const stopKeyboardPropagation = useCallback((event: KeyboardEvent) => event.stopPropagation(), []);

  const handleAutoNegative = useCallback(
    (details: { checked: boolean }) => {
      commitConfig(
        t('widgets.layers.regionalGuidance.autoNegative'),
        { autoNegative: details.checked },
        { autoNegative: layer.autoNegative }
      );
    },
    [commitConfig, layer.autoNegative, t]
  );

  const styleCollection = useMemo(
    () =>
      createListCollection({
        items: MASK_FILL_STYLES.map((style) => ({
          label: t(`widgets.layers.maskFill.styles.${style}`),
          value: style,
        })),
      }),
    [t]
  );

  const commitFill = useCallback(
    (next: CanvasMaskFillContract, before: CanvasMaskFillContract) => {
      commitConfig(t('widgets.layers.maskFill.fill'), { mask: { fill: next } }, { mask: { fill: before } });
    },
    [commitConfig, t]
  );

  const handleColorChange = useCallback(
    (hex: string) => {
      if (
        !applyStructuralPreview(engine, dispatch, {
          config: { layerType: 'regional_guidance', mask: { fill: { ...fill, color: hex } } },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        })
      ) {
        return;
      }
      if (fillBeforeRef.current === null) {
        fillBeforeRef.current = fill;
      }
    },
    [dispatch, engine, fill, layer.id]
  );

  const handleColorChangeEnd = useCallback(
    (hex: string) => {
      const before = fillBeforeRef.current ?? fill;
      fillBeforeRef.current = null;
      commitFill({ ...before, color: hex }, before);
    },
    [commitFill, fill]
  );

  const handleStyleChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const style = value[0] as CanvasMaskFillContract['style'] | undefined;
      if (style && style !== fill.style) {
        commitFill({ ...fill, style }, fill);
      }
    },
    [commitFill, fill]
  );

  const handleInvert = useCallback(() => {
    engine?.layers.invertMask(layer.id);
  }, [engine, layer.id]);

  // Reference-image helpers ---------------------------------------------------
  const referenceImages = layer.referenceImages;

  const commitReferenceImages = useCallback(
    (next: RegionalGuidanceReferenceImage[]) => {
      commitConfig(
        t('widgets.layers.regionalGuidance.referenceImages'),
        { referenceImages: next },
        {
          referenceImages: referenceImages,
        }
      );
    },
    [commitConfig, referenceImages, t]
  );

  const handleAddReferenceImage = useCallback(() => {
    commitReferenceImages([...referenceImages, createReferenceImage(base)]);
  }, [base, commitReferenceImages, referenceImages]);

  // Assigning the image goes through the same undo path as the other ref edits
  // (`commitReferenceImages` → `updateCanvasLayerConfig`), so a drop/upload/clear
  // is a single, undoable document change.
  const setReferenceImageAsset = useCallback(
    (refId: string, image: RegionalGuidanceReferenceImageAsset | null) => {
      commitReferenceImages(
        referenceImages.map((ref) => (ref.id === refId ? { ...ref, config: { ...ref.config, image } } : ref))
      );
    },
    [commitReferenceImages, referenceImages]
  );

  const uploadReferenceImageAsset = useCallback(
    async (refId: string, file: File) => {
      try {
        const uploaded = await galleryTransfers.upload(file, 'none');
        setReferenceImageAsset(refId, uploaded);
        gallery.touchImages();
      } catch (error) {
        notifications.reportError({
          area: 'regional-guidance',
          message: error instanceof Error ? error.message : String(error),
          namespace: 'generation',
        });
      }
    },
    [gallery, notifications, setReferenceImageAsset]
  );

  // A single monitor routes gallery-image drops to the region's ref slot the drop
  // landed on (each row registers a droppable keyed by layer + ref id).
  useDndMonitor({
    onDragEnd: (event) => {
      const overId = event.over?.id;
      const prefix = `regional-ref-image:${layer.id}:`;
      if (typeof overId !== 'string' || !overId.startsWith(prefix)) {
        return;
      }
      const data = event.active.data.current;
      if (!isGalleryImageDragData(data) || data.images.length === 0) {
        return;
      }
      const refId = overId.slice(prefix.length);
      const [first] = data.images;
      void galleryImages.resolveMany([first.imageName]).then((images) => {
        if (images[0]) {
          setReferenceImageAsset(refId, images[0]);
        }
      });
    },
  });

  const ipAdapterModelCollection = useMemo(
    () =>
      createListCollection({
        items: models
          .filter((model) => model.type === 'ip_adapter' && (!base || model.base === base))
          .map((model) => ({ label: model.name, value: model.key })),
      }),
    [base, models]
  );

  const fluxReduxModelCollection = useMemo(
    () =>
      createListCollection({
        items: models
          .filter((model) => model.type === 'flux_redux' && (!base || model.base === base))
          .map((model) => ({ label: model.name, value: model.key })),
      }),
    [base, models]
  );

  const methodCollection = useMemo(
    () =>
      createListCollection({
        items: IP_ADAPTER_METHODS.map((method) => ({
          label: t(`widgets.layers.regionalGuidance.methods.${method}`),
          value: method as string,
        })),
      }),
    [t]
  );

  const styleValue = useMemo(() => [fill.style], [fill.style]);

  return (
    <Stack gap="2" onKeyDown={stopKeyboardPropagation}>
      <Field label={t('widgets.layers.regionalGuidance.positivePrompt')}>
        <PromptTextarea
          {...PROMPT_ATTENTION_TARGET_PROPS}
          aria-label={t('widgets.layers.regionalGuidance.positivePrompt')}
          defaultHeightPx={REGIONAL_PROMPT_HEIGHT_PX}
          minHeightPx={REGIONAL_PROMPT_HEIGHT_PX}
          placeholder={t('widgets.layers.regionalGuidance.positivePromptPlaceholder')}
          resizeHandleAriaLabel={t('widgets.layers.regionalGuidance.positivePrompt')}
          showSyntaxHighlighting={showSyntaxHighlighting}
          size="sm"
          value={positivePrompt}
          onBlur={handlePositiveBlur}
          onChange={handlePositiveChange}
        />
      </Field>
      {!isFlux && (
        <Field label={t('widgets.layers.regionalGuidance.negativePrompt')}>
          <PromptTextarea
            {...PROMPT_ATTENTION_TARGET_PROPS}
            aria-label={t('widgets.layers.regionalGuidance.negativePrompt')}
            defaultHeightPx={REGIONAL_PROMPT_HEIGHT_PX}
            minHeightPx={REGIONAL_PROMPT_HEIGHT_PX}
            placeholder={t('widgets.layers.regionalGuidance.negativePromptPlaceholder')}
            resizeHandleAriaLabel={t('widgets.layers.regionalGuidance.negativePrompt')}
            showSyntaxHighlighting={showSyntaxHighlighting}
            size="sm"
            value={negativePrompt}
            onBlur={handleNegativeBlur}
            onChange={handleNegativeChange}
          />
        </Field>
      )}
      {!isFlux && (
        <Switch.Root checked={layer.autoNegative} size="sm" onCheckedChange={handleAutoNegative}>
          <Switch.HiddenInput />
          <Switch.Control>
            <Switch.Thumb />
          </Switch.Control>
          <Switch.Label>
            <Text fontSize="xs">{t('widgets.layers.regionalGuidance.autoNegative')}</Text>
          </Switch.Label>
        </Switch.Root>
      )}

      <HStack gap="2">
        <Field flexShrink="0" label={t('widgets.layers.maskFill.color')}>
          <ColorPicker
            aria-label={t('widgets.layers.maskFill.color')}
            value={fill.color}
            onValueChange={handleColorChange}
            onValueChangeEnd={handleColorChangeEnd}
          />
        </Field>
        <Field flex="1" label={t('widgets.layers.maskFill.style')} minW="0">
          <Select
            aria-label={t('widgets.layers.maskFill.style')}
            collection={styleCollection}
            positioning={SELECT_POSITIONING}
            size="xs"
            value={styleValue}
            valueText={t(`widgets.layers.maskFill.styles.${fill.style}`)}
            onValueChange={handleStyleChange}
          />
        </Field>
      </HStack>
      <Button disabled={!engine} size="xs" variant="outline" onClick={handleInvert}>
        {t('widgets.layers.maskFill.invert')}
      </Button>

      <Stack gap="2">
        <HStack justify="space-between">
          <Text fontSize="xs" fontWeight="medium">
            {t('widgets.layers.regionalGuidance.referenceImages')}
          </Text>
          <IconButton
            aria-label={t('widgets.layers.regionalGuidance.addReferenceImage')}
            size="2xs"
            variant="ghost"
            onClick={handleAddReferenceImage}
          >
            <PlusIcon />
          </IconButton>
        </HStack>
        {referenceImages.map((ref, index) => (
          <ReferenceImageRow
            dropId={referenceImageDropId(layer.id, ref.id)}
            index={index}
            ipAdapterModelCollection={ipAdapterModelCollection}
            fluxReduxModelCollection={fluxReduxModelCollection}
            key={ref.id}
            methodCollection={methodCollection}
            models={models}
            referenceImage={ref}
            referenceImages={referenceImages}
            onCommit={commitReferenceImages}
            onSetImage={setReferenceImageAsset}
            onUpload={uploadReferenceImageAsset}
          />
        ))}
      </Stack>
    </Stack>
  );
};

interface ReferenceImageRowProps {
  dropId: string;
  index: number;
  referenceImage: RegionalGuidanceReferenceImage;
  referenceImages: readonly RegionalGuidanceReferenceImage[];
  ipAdapterModelCollection: ReturnType<typeof createListCollection<{ label: string; value: string }>>;
  fluxReduxModelCollection: ReturnType<typeof createListCollection<{ label: string; value: string }>>;
  methodCollection: ReturnType<typeof createListCollection<{ label: string; value: string }>>;
  models: readonly ModelConfig[];
  onCommit: (next: RegionalGuidanceReferenceImage[]) => void;
  onSetImage: (refId: string, image: RegionalGuidanceReferenceImageAsset | null) => void;
  onUpload: (refId: string, file: File) => void;
}

/**
 * One regional reference image: enable/remove, an image slot (gallery drop or
 * upload, mirroring the generate widget's mechanism), and the model + adapter
 * fields for its config kind — IP-Adapter (model/method/weight) for non-FLUX
 * regions, FLUX Redux (model/influence) for FLUX regions.
 */
const ReferenceImageRow = ({
  dropId,
  index,
  ipAdapterModelCollection,
  fluxReduxModelCollection,
  methodCollection,
  models,
  referenceImage,
  referenceImages,
  onCommit,
  onSetImage,
  onUpload,
}: ReferenceImageRowProps) => {
  const { t } = useTranslation();
  const { config } = referenceImage;
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { isOver, setNodeRef } = useDroppable({ data: { kind: 'regional-reference-image' }, id: dropId });

  const replaceRef = useCallback(
    (next: RegionalGuidanceReferenceImage) => {
      onCommit(referenceImages.map((entry, i) => (i === index ? next : entry)));
    },
    [index, onCommit, referenceImages]
  );

  const handleRemove = useCallback(() => {
    onCommit(referenceImages.filter((_, i) => i !== index));
  }, [index, onCommit, referenceImages]);

  const handleEnabled = useCallback(
    (details: { checked: boolean }) => {
      replaceRef({ ...referenceImage, isEnabled: details.checked });
    },
    [referenceImage, replaceRef]
  );

  const openUpload = useCallback(() => fileInputRef.current?.click(), []);

  const handleFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];
      if (file) {
        onUpload(referenceImage.id, file);
      }
      event.currentTarget.value = '';
    },
    [onUpload, referenceImage.id]
  );

  const handleClearImage = useCallback(() => onSetImage(referenceImage.id, null), [onSetImage, referenceImage.id]);

  const handleModel = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      if (config.type !== 'ip_adapter' && config.type !== 'flux_redux') {
        return;
      }
      const key = value[0];
      const model = models.find((entry) => entry.key === key) ?? null;
      const modelIdentity = model ? { base: model.base, key: model.key, name: model.name, type: model.type } : null;
      replaceRef({ ...referenceImage, config: { ...config, model: modelIdentity } });
    },
    [config, models, referenceImage, replaceRef]
  );

  const handleMethod = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      if (config.type !== 'ip_adapter') {
        return;
      }
      const method = value[0] as RegionalGuidanceIPAdapterMethod | undefined;
      if (method) {
        replaceRef({ ...referenceImage, config: { ...config, method } });
      }
    },
    [config, referenceImage, replaceRef]
  );

  // Local live value while dragging the weight slider — the thumb tracks the
  // pointer without pushing a history entry / re-committing the ref per tick. The
  // single commit lands on drag end. `null` ⇒ not dragging (show the stored value).
  const [liveWeight, setLiveWeight] = useState<number | null>(null);

  const handleWeight = useCallback(({ value }: SliderValueChangeDetails) => {
    const next = value[0];
    if (next === undefined || !Number.isFinite(next)) {
      return;
    }
    setLiveWeight(next);
  }, []);

  const handleWeightEnd = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      const next = value[0];
      setLiveWeight(null);
      if (config.type !== 'ip_adapter' || next === undefined || !Number.isFinite(next)) {
        return;
      }
      replaceRef({ ...referenceImage, config: { ...config, weight: next } });
    },
    [config, referenceImage, replaceRef]
  );

  const handleFluxReduxConfig = useCallback(
    (nextConfig: RegionalGuidanceReferenceImage['config']) => {
      replaceRef({ ...referenceImage, config: nextConfig });
    },
    [referenceImage, replaceRef]
  );

  const modelCollection = config.type === 'flux_redux' ? fluxReduxModelCollection : ipAdapterModelCollection;
  const modelValue = useMemo(
    () =>
      config.type === 'ip_adapter' || config.type === 'flux_redux' ? (config.model ? [config.model.key] : []) : [],
    [config]
  );
  const methodValue = useMemo(() => (config.type === 'ip_adapter' ? [config.method] : []), [config]);
  const weightValue = useMemo(
    () => (liveWeight !== null ? [liveWeight] : config.type === 'ip_adapter' ? [config.weight] : [1]),
    [config, liveWeight]
  );
  const weightAria = useMemo(() => [t('widgets.layers.regionalGuidance.weight')], [t]);

  const image = config.image;
  const modelName =
    'model' in config && config.model ? config.model.name : t('widgets.layers.regionalGuidance.selectModel');

  return (
    <Stack borderColor="border.subtle" borderWidth="1px" gap="2" p="2" rounded="md">
      <HStack justify="space-between">
        <Switch.Root checked={referenceImage.isEnabled} size="sm" onCheckedChange={handleEnabled}>
          <Switch.HiddenInput />
          <Switch.Control>
            <Switch.Thumb />
          </Switch.Control>
          <Switch.Label>
            <Text fontSize="xs">{`${t('widgets.layers.regionalGuidance.referenceImage')} ${index + 1}`}</Text>
          </Switch.Label>
        </Switch.Root>
        <IconButton
          aria-label={t('widgets.layers.regionalGuidance.removeReferenceImage')}
          size="2xs"
          variant="ghost"
          onClick={handleRemove}
        >
          <XIcon />
        </IconButton>
      </HStack>

      <HStack align="flex-start" gap="2">
        <Box
          bg={isOver ? 'accent.muted' : undefined}
          borderColor={isOver ? 'accent.solid' : 'border.emphasized'}
          borderStyle={image ? 'solid' : 'dashed'}
          borderWidth="1px"
          flexShrink="0"
          h="16"
          overflow="hidden"
          position="relative"
          ref={setNodeRef}
          rounded="md"
          w="16"
        >
          <Box
            as="button"
            alignItems="center"
            aria-label={t('widgets.layers.regionalGuidance.setReferenceImage')}
            bg="bg.muted"
            color="fg.muted"
            cursor="pointer"
            display="flex"
            h="full"
            justifyContent="center"
            w="full"
            onClick={openUpload}
          >
            {image ? (
              <img alt={image.imageName} draggable={false} src={image.thumbnailUrl} style={COVER_IMG_STYLE} />
            ) : (
              <ImageIcon size="20" />
            )}
          </Box>
          {image ? (
            <IconButton
              aria-label={t('widgets.layers.regionalGuidance.clearReferenceImage')}
              colorPalette="red"
              position="absolute"
              right="0.5"
              size="2xs"
              top="0.5"
              variant="solid"
              onClick={handleClearImage}
            >
              <XIcon />
            </IconButton>
          ) : null}
        </Box>
        <Text color="fg.muted" flex="1" fontSize="2xs" minW="0">
          {t('widgets.layers.regionalGuidance.referenceImageHelp')}
        </Text>
        <Input ref={fileInputRef} accept="image/*" display="none" type="file" onChange={handleFileChange} />
      </HStack>

      <Field label={t('widgets.layers.regionalGuidance.model')}>
        <Select
          aria-label={t('widgets.layers.regionalGuidance.model')}
          collection={modelCollection}
          positioning={SELECT_POSITIONING}
          size="xs"
          value={modelValue}
          valueText={modelName}
          onValueChange={handleModel}
        />
      </Field>

      {config.type === 'ip_adapter' ? (
        <>
          <Field label={t('widgets.layers.regionalGuidance.method')}>
            <Select
              aria-label={t('widgets.layers.regionalGuidance.method')}
              collection={methodCollection}
              positioning={SELECT_POSITIONING}
              size="xs"
              value={methodValue}
              valueText={t(`widgets.layers.regionalGuidance.methods.${config.method}`)}
              onValueChange={handleMethod}
            />
          </Field>
          <Field label={t('widgets.layers.regionalGuidance.weight')}>
            <Slider
              aria-label={weightAria}
              formatValue={formatWeight}
              max={2}
              min={-1}
              size="sm"
              step={0.01}
              value={weightValue}
              withThumbTooltip
              onValueChange={handleWeight}
              onValueChangeEnd={handleWeightEnd}
            />
          </Field>
        </>
      ) : config.type === 'flux_redux' ? (
        <FluxReduxControls config={config} disabled={false} onChange={handleFluxReduxConfig} />
      ) : null}
    </Stack>
  );
};

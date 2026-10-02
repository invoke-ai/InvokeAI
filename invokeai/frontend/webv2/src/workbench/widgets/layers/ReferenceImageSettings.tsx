import type { SelectValueChangeDetails, SliderValueChangeDetails } from '@chakra-ui/react';
import type { ModelConfig } from '@features/models';
import type {
  CanvasRegionalGuidanceLayerContract,
  RegionalGuidanceIPAdapterMethod,
  RegionalGuidanceReferenceImage,
  RegionalGuidanceReferenceImageAsset,
} from '@workbench/canvas-engine/api';
import type { ChangeEvent, CSSProperties } from 'react';

import { Box, createListCollection, HStack, IconButton, Input, Stack, Text } from '@chakra-ui/react';
import { useDndMonitor } from '@dnd-kit/core';
import { galleryImageItemToGalleryImage, galleryImages, galleryTransfers, type GalleryItem } from '@features/gallery';
import { GalleryPickerPopover } from '@features/gallery/picker';
import { invalidateGallery } from '@features/gallery/queries';
import { isGalleryImageDragData, useGalleryImageDroppable } from '@features/gallery/utility';
import { FluxReduxControls } from '@features/generation/components';
import { useModelsSelector } from '@features/models';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { Button, DropZone, Field, Select, Slider } from '@platform/ui';
import { useQueryClient } from '@tanstack/react-query';
import { type CanvasPreparedEngine, usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { ImageIcon, UploadIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useSelectedModelBase } from './useSelectedModelBase';

/**
 * Edit regional references through whole-array patch-config; tree rows remain the sole inventory, enable, and
 * removal controls.
 */

const IP_ADAPTER_METHODS: readonly RegionalGuidanceIPAdapterMethod[] = [
  'full',
  'style',
  'composition',
  'style_strong',
  'style_precise',
];

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: true } as const;

const COVER_IMG_STYLE: CSSProperties = { height: '100%', objectFit: 'cover', width: '100%' };
const IMAGE_ONLY = ['image'] as const;

const formatWeight = (value: number): string => value.toFixed(2);

/** DnD droppable id for a specific region's reference-image slot (all-image gallery-item drop target). */
export const referenceImageDropId = (layerId: string, refId: string): string =>
  `regional-ref-image:${layerId}:${refId}`;

interface ReferenceImageEditing {
  commitReferenceImages(next: RegionalGuidanceReferenceImage[]): void;
  setReferenceImageAsset(refId: string, image: RegionalGuidanceReferenceImageAsset | null): void;
  uploadReferenceImageAsset(refId: string, file: File): void;
}

const useReferenceImageEditing = (
  engine: CanvasPreparedEngine | null,
  layer: CanvasRegionalGuidanceLayerContract
): ReferenceImageEditing => {
  const { t } = useTranslation();
  const commitPrepared = usePreparedCommit(engine);
  const { notifications } = useWorkbenchCommands();
  const queryClient = useQueryClient();
  const referenceImages = layer.referenceImages;

  const commitReferenceImages = useCallback(
    (next: RegionalGuidanceReferenceImage[]) => {
      commitPrepared(t('widgets.layers.regionalGuidance.referenceImages'), (model) =>
        model.prepare({
          before: { layerType: 'regional_guidance', referenceImages: [...referenceImages] },
          config: { layerType: 'regional_guidance', referenceImages: next },
          id: layer.id,
          type: 'patch-config',
        })
      );
    },
    [commitPrepared, layer.id, referenceImages, t]
  );

  const setReferenceImageAsset = useCallback(
    (refId: string, image: RegionalGuidanceReferenceImageAsset | null) => {
      commitReferenceImages(
        referenceImages.map((ref) => (ref.id === refId ? { ...ref, config: { ...ref.config, image } } : ref))
      );
    },
    [commitReferenceImages, referenceImages]
  );

  const uploadReferenceImageAsset = useCallback(
    (refId: string, file: File) => {
      const owner = captureAccountScope();
      void (async () => {
        try {
          const uploaded = await galleryTransfers.upload(file, 'none', { signal: owner.signal });
          assertAccountScopeCurrent(owner);
          setReferenceImageAsset(refId, uploaded);
          void invalidateGallery(queryClient, owner);
        } catch (error) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }
          notifications.reportError({
            area: 'regional-guidance',
            message: error instanceof Error ? error.message : String(error),
            namespace: 'generation',
          });
        }
      })();
    },
    [notifications, queryClient, setReferenceImageAsset]
  );

  useDndMonitor({
    onDragEnd: (event) => {
      const overId = event.over?.id;
      const prefix = `regional-ref-image:${layer.id}:`;
      if (typeof overId !== 'string' || !overId.startsWith(prefix)) {
        return;
      }
      const data = event.active.data.current;
      if (!isGalleryImageDragData(data)) {
        return;
      }
      const refId = overId.slice(prefix.length);
      const [first] = data.items;
      void galleryImages.resolveMany([first.name]).then((images) => {
        if (images[0]) {
          setReferenceImageAsset(refId, images[0]);
        }
      });
    },
  });

  return useMemo(
    () => ({ commitReferenceImages, setReferenceImageAsset, uploadReferenceImageAsset }),
    [commitReferenceImages, setReferenceImageAsset, uploadReferenceImageAsset]
  );
};

type ModelCollection = ReturnType<typeof createListCollection<{ label: string; value: string }>>;

interface ReferenceImageCollections {
  fluxRedux: ModelCollection;
  ipAdapter: ModelCollection;
  method: ModelCollection;
  models: readonly ModelConfig[];
}

/** The model/method option collections the editor selects from. */
const useReferenceImageCollections = (): ReferenceImageCollections => {
  const { t } = useTranslation();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const base = useSelectedModelBase();

  const ipAdapter = useMemo(
    () =>
      createListCollection({
        items: models
          .filter((model) => model.type === 'ip_adapter' && (!base || model.base === base))
          .map((model) => ({ label: model.name, value: model.key })),
      }),
    [base, models]
  );
  const fluxRedux = useMemo(
    () =>
      createListCollection({
        items: models
          .filter((model) => model.type === 'flux_redux' && (!base || model.base === base))
          .map((model) => ({ label: model.name, value: model.key })),
      }),
    [base, models]
  );
  const method = useMemo(
    () =>
      createListCollection({
        items: IP_ADAPTER_METHODS.map((value) => ({
          label: t(`widgets.layers.regionalGuidance.methods.${value}`),
          value: value as string,
        })),
      }),
    [t]
  );

  return useMemo(() => ({ fluxRedux, ipAdapter, method, models }), [fluxRedux, ipAdapter, method, models]);
};

/**
 * Resolves the sub-selected reference image; renders nothing once the item is
 * gone — the sub-selection reconciler hands the pane back to the layer.
 */
export const ReferenceImageSettings = ({
  engine,
  layer,
  refId,
}: {
  engine: CanvasPreparedEngine | null;
  layer: CanvasRegionalGuidanceLayerContract;
  refId: string;
}) => {
  const index = layer.referenceImages.findIndex((ref) => ref.id === refId);
  const referenceImage = layer.referenceImages[index];
  if (!referenceImage) {
    return null;
  }
  return <ReferenceImageEditor engine={engine} index={index} layer={layer} referenceImage={referenceImage} />;
};

const ReferenceImageEditor = ({
  engine,
  index,
  layer,
  referenceImage,
}: {
  engine: CanvasPreparedEngine | null;
  index: number;
  layer: CanvasRegionalGuidanceLayerContract;
  referenceImage: RegionalGuidanceReferenceImage;
}) => {
  const { t } = useTranslation();
  const editing = useReferenceImageEditing(engine, layer);
  const collections = useReferenceImageCollections();
  const { config } = referenceImage;
  const referenceImages = layer.referenceImages;
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { isOver, setNodeRef } = useGalleryImageDroppable({
    data: { kind: 'regional-reference-image' },
    id: referenceImageDropId(layer.id, referenceImage.id),
  });

  const replaceRef = useCallback(
    (next: RegionalGuidanceReferenceImage) => {
      editing.commitReferenceImages(referenceImages.map((entry, i) => (i === index ? next : entry)));
    },
    [editing, index, referenceImages]
  );

  const openUpload = useCallback(() => fileInputRef.current?.click(), []);

  const handleFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];
      if (file) {
        editing.uploadReferenceImageAsset(referenceImage.id, file);
      }
      event.currentTarget.value = '';
    },
    [editing, referenceImage.id]
  );

  const handleClearImage = useCallback(
    () => editing.setReferenceImageAsset(referenceImage.id, null),
    [editing, referenceImage.id]
  );

  const handlePick = useCallback(
    (item: GalleryItem) => {
      if (item.kind === 'image') {
        editing.setReferenceImageAsset(referenceImage.id, galleryImageItemToGalleryImage(item));
      }
    },
    [editing, referenceImage.id]
  );

  const handleModel = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      if (config.type !== 'ip_adapter' && config.type !== 'flux_redux') {
        return;
      }
      const key = value[0];
      const model = collections.models.find((entry) => entry.key === key) ?? null;
      const modelIdentity = model ? { base: model.base, key: model.key, name: model.name, type: model.type } : null;
      replaceRef({ ...referenceImage, config: { ...config, model: modelIdentity } });
    },
    [collections.models, config, referenceImage, replaceRef]
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

  const modelCollection = config.type === 'flux_redux' ? collections.fluxRedux : collections.ipAdapter;
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
    <Stack gap="2">
      <HStack align="flex-start" gap="2">
        <DropZone
          ref={setNodeRef}
          borderStyle={image ? 'solid' : 'dashed'}
          flexShrink="0"
          h="16"
          isOver={isOver}
          overflow="hidden"
          position="relative"
          w="16"
        >
          <GalleryPickerPopover
            accept={IMAGE_ONLY}
            label={t(image ? 'widgets.gallery.picker.replaceImage' : 'widgets.gallery.picker.chooseImage')}
            onPick={handlePick}
          >
            <Box
              as="button"
              alignItems="center"
              aria-label={t(image ? 'widgets.gallery.picker.replaceImage' : 'widgets.gallery.picker.chooseImage')}
              bg="bg.muted"
              color="fg.muted"
              display="flex"
              h="full"
              justifyContent="center"
              w="full"
            >
              {image ? (
                <img alt={image.imageName} draggable={false} src={image.thumbnailUrl} style={COVER_IMG_STYLE} />
              ) : (
                <ImageIcon size="20" />
              )}
            </Box>
          </GalleryPickerPopover>
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
        </DropZone>
        <Stack align="start" flex="1" gap="1" minW="0">
          <Text color="fg.muted" fontSize="2xs">
            {t('widgets.layers.regionalGuidance.referenceImageHelp')}
          </Text>
          <Button size="xs" variant="ghost" onClick={openUpload}>
            <UploadIcon size="12" />
            {t('widgets.gallery.picker.upload')}
          </Button>
        </Stack>
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
              collection={collections.method}
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

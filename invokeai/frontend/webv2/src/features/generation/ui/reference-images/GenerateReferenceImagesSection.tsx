import type { GalleryItem } from '@features/gallery';
import type { GalleryPickerSelection } from '@features/gallery/picker';
import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type {
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateSettings,
} from '@features/generation/core/types';
import type { GenerateSettingsUpdate } from '@features/generation/ui/generateDebounce';

import { HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { useDndMonitor } from '@dnd-kit/core';
import { galleryImages, toGalleryItemKey } from '@features/gallery';
import { GalleryPickerPopover } from '@features/gallery/picker';
import { isGalleryImageDragData, useGalleryImageDroppable } from '@features/gallery/utility';
import {
  createReferenceImageId,
  getDefaultReferenceImageConfig,
  getGenerationDimensions,
  getMaxReferenceImages,
  isReferenceImageSupported,
} from '@features/generation/core/baseGenerationPolicies';
import { generatedImageToReferenceImage, getEffectiveReferenceImage } from '@features/generation/core/referenceImage';
import { clampDimension, deriveAspectRatioId, moveReferenceImage } from '@features/generation/core/settings';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { Button, DropTargetOverlay, DropZone } from '@platform/ui';
import { ChevronDownIcon, ImagePlusIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { ReferenceImageCard } from './ReferenceImageCard';

const UPLOAD_ZONE_HOVER_STYLES = { bg: 'bg.muted', color: 'fg' };
const IMAGE_ONLY = ['image'] as const;

interface GenerateReferenceImagesContentProps {
  models: readonly ModelConfig[];
  selectedModel: GenerateModelConfig | undefined;
  settings: GenerateSettings;
  onCommit: (update: GenerateSettingsUpdate) => void;
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

export const GenerateReferenceImagesContent = ({
  models,
  onCommit,
  onCommitImmediate,
  selectedModel,
  settings,
}: GenerateReferenceImagesContentProps) => {
  const { t } = useTranslation();
  const { gallery } = useGenerationUi();
  const referenceImages = settings.referenceImages;
  const isSupported = isReferenceImageSupported(selectedModel);
  const maxReferenceImages = getMaxReferenceImages(selectedModel);
  const canAdd = referenceImages.length < maxReferenceImages;
  const { acceptsActiveDrag, isOver, setNodeRef } = useGalleryImageDroppable({
    data: { kind: 'generate-reference-images' },
    disabled: !canAdd,
    id: 'generate-reference-images',
  });
  const referenceImageCount = referenceImages.length;

  const appendReferenceImages = useCallback(
    (images: GenerateReferenceImageAsset[]) => {
      // Mint IDs outside replayed updaters so draft and flushed identities match.
      const ids = images.map(() => createReferenceImageId());

      onCommit((currentSettings) => {
        const remaining = getMaxReferenceImages(selectedModel) - currentSettings.referenceImages.length;

        if (remaining <= 0 || images.length === 0) {
          return currentSettings;
        }

        return {
          ...currentSettings,
          referenceImages: [
            ...currentSettings.referenceImages,
            ...images.slice(0, remaining).map((image, index) => ({
              config: getDefaultReferenceImageConfig(selectedModel, models, image),
              id: ids[index] ?? createReferenceImageId(),
              isEnabled: true,
            })),
          ],
        };
      });
    },
    [models, onCommit, selectedModel]
  );

  const patchReferenceImage = useCallback(
    (id: string, patch: Partial<GenerateReferenceImage>) => {
      onCommit((currentSettings) => ({
        ...currentSettings,
        referenceImages: currentSettings.referenceImages.map((referenceImage) =>
          referenceImage.id === id ? { ...referenceImage, ...patch } : referenceImage
        ),
      }));
    },
    [onCommit]
  );

  const removeReferenceImage = useCallback(
    (id: string) => {
      onCommit((currentSettings) => ({
        ...currentSettings,
        referenceImages: currentSettings.referenceImages.filter((referenceImage) => referenceImage.id !== id),
      }));
    },
    [onCommit]
  );

  // Use functional reorder updates to preserve concurrent debounced changes.
  const handleMoveReferenceImage = useCallback(
    (id: string, direction: -1 | 1) => {
      onCommit((currentSettings) => {
        const referenceImages = moveReferenceImage(currentSettings.referenceImages, id, direction);

        return referenceImages === currentSettings.referenceImages
          ? currentSettings
          : { ...currentSettings, referenceImages: [...referenceImages] };
      });
    },
    [onCommit]
  );

  const clearReferenceImages = useCallback(() => onCommitImmediate({ referenceImages: [] }), [onCommitImmediate]);

  const pickerSelection = useMemo<GalleryPickerSelection>(
    () => ({
      addedKeys: new Set(
        referenceImages.flatMap(({ config }) =>
          config.image ? [toGalleryItemKey({ kind: 'image', name: config.image.original.image.image_name })] : []
        )
      ),
      mode: 'multiple',
      remaining: { image: Math.max(0, maxReferenceImages - referenceImageCount) },
    }),
    [maxReferenceImages, referenceImageCount, referenceImages]
  );

  const handlePick = useCallback(
    (item: GalleryItem) => {
      if (item.kind === 'image') {
        appendReferenceImages([
          generatedImageToReferenceImage({ height: item.height, imageName: item.name, width: item.width }),
        ]);
      }
    },
    [appendReferenceImages]
  );

  const applyReferenceImageSize = useCallback(
    (image: GenerateReferenceImageAsset) => {
      if (!selectedModel) {
        return;
      }

      const grid = getGenerationDimensions(selectedModel).grid;
      const effectiveImage = getEffectiveReferenceImage(image);
      const width = clampDimension(effectiveImage.width, grid);
      const height = clampDimension(effectiveImage.height, grid);

      onCommitImmediate({
        aspectRatioId: deriveAspectRatioId(width, height),
        aspectRatioValue: height > 0 ? width / height : 1,
        height,
        width,
      });
    },
    [onCommitImmediate, selectedModel]
  );

  const addGalleryImages = async (imageNames: string[]) => {
    if (!canAdd || imageNames.length === 0) {
      return;
    }

    const images = await galleryImages.resolveMany(imageNames.slice(0, maxReferenceImages - referenceImageCount));

    if (images.length > 0) {
      appendReferenceImages(images.map(generatedImageToReferenceImage));
    }
  };

  useDndMonitor({
    onDragEnd: (event) => {
      const data = event.active.data.current;

      if (event.over?.id === 'generate-reference-images' && isGalleryImageDragData(data) && canAdd) {
        void addGalleryImages(data.items.map((item) => item.name));
      }
    },
  });

  if (!isSupported && referenceImageCount === 0) {
    return null;
  }

  // Unsupported-model references remain read-only with removal as recovery.
  if (!isSupported) {
    return (
      <HStack gap="2" justify="space-between">
        <Text color="fg.muted" fontSize="2xs" minW="0">
          {t('widgets.generate.referenceImagesUnsupported')}
        </Text>
        <Button colorPalette="red" flexShrink="0" size="xs" variant="outline" onClick={clearReferenceImages}>
          {t('widgets.generate.clearReferenceImages')}
        </Button>
      </HStack>
    );
  }

  return (
    <Stack ref={setNodeRef} gap="2" position="relative">
      <DropTargetOverlay
        isActive={acceptsActiveDrag}
        isOver={isOver}
        label={t('widgets.generate.dropReferenceImages')}
      />
      <GalleryPickerPopover
        accept={IMAGE_ONLY}
        label={t('widgets.generate.addReferenceImage')}
        selection={pickerSelection}
        onPick={handlePick}
      >
        <DropZone
          as="button"
          alignItems="center"
          cursor={canAdd ? undefined : 'not-allowed'}
          disabled={!canAdd}
          display="flex"
          flexDirection="column"
          fontSize="2xs"
          gap="1"
          isDisabled={!canAdd}
          isOver={isOver}
          justifyContent="center"
          minH="14"
          minW="0"
          opacity={canAdd ? 1 : 0.6}
          px="3"
          w="full"
          _hover={canAdd ? UPLOAD_ZONE_HOVER_STYLES : undefined}
        >
          <HStack as="span" color="fg" fontSize="xs" fontWeight="600" gap="1.5">
            <Icon as={ImagePlusIcon} boxSize="4" />
            {t('widgets.generate.addReferenceImage')}
            <Icon as={ChevronDownIcon} boxSize="3" color="fg.subtle" />
          </HStack>
          <Text as="span" color="fg.muted">
            {t('widgets.gallery.picker.dropHint')}
          </Text>
        </DropZone>
      </GalleryPickerPopover>

      {referenceImageCount > 0 ? (
        <Stack gap="2">
          {referenceImages.map((referenceImage, index) => (
            <ReferenceImageCard
              key={referenceImage.id}
              count={referenceImageCount}
              index={index}
              referenceImage={referenceImage}
              selectedModel={selectedModel}
              onFindInGallery={gallery.findImage}
              onMove={handleMoveReferenceImage}
              onPatch={patchReferenceImage}
              onRemove={removeReferenceImage}
              onUseSize={applyReferenceImageSize}
            />
          ))}
        </Stack>
      ) : null}
    </Stack>
  );
};

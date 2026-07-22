import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type {
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateSettings,
} from '@features/generation/core/types';
import type { GenerateSettingsUpdate } from '@features/generation/ui/generateDebounce';
import type { ChangeEvent } from 'react';

import { Badge, HStack, Input, Stack, Text } from '@chakra-ui/react';
import { useDndMonitor, useDroppable } from '@dnd-kit/core';
import { galleryImages, galleryTransfers } from '@features/gallery';
import { isGalleryImageDragData } from '@features/gallery/utility';
import {
  createReferenceImageId,
  getDefaultReferenceImageConfig,
  getGenerationDimensions,
  getMaxReferenceImages,
  isReferenceImageSupported,
} from '@features/generation/core/baseGenerationPolicies';
import { generatedImageToReferenceImage, getEffectiveReferenceImage } from '@features/generation/core/referenceImage';
import { clampDimension, deriveAspectRatioId } from '@features/generation/core/settings';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { GenerateCollapsibleSection } from '@features/generation/ui/shared/GenerateCollapsibleSection';
import { Button, DropZone } from '@platform/ui';
import { UploadIcon } from 'lucide-react';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { ReferenceImageCard } from './ReferenceImageCard';

const UPLOAD_ZONE_HOVER_STYLES = { bg: 'bg.muted', color: 'fg' };

interface GenerateReferenceImagesSectionProps {
  models: readonly ModelConfig[];
  selectedModel: GenerateModelConfig | undefined;
  settings: GenerateSettings;
  onCommit: (update: GenerateSettingsUpdate) => void;
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

export const GenerateReferenceImagesSection = ({
  models,
  onCommit,
  onCommitImmediate,
  selectedModel,
  settings,
}: GenerateReferenceImagesSectionProps) => {
  const { t } = useTranslation();
  const { gallery, notifications } = useGenerationUi();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const referenceImages = settings.referenceImages;
  const isSupported = isReferenceImageSupported(selectedModel);
  const maxReferenceImages = getMaxReferenceImages(selectedModel);
  const canAdd = referenceImages.length < maxReferenceImages;
  const { isOver, setNodeRef } = useDroppable({
    data: { kind: 'generate-reference-images' },
    disabled: !canAdd,
    id: 'generate-reference-images',
  });
  const activeCount = referenceImages.filter((image) => image.isEnabled).length;
  const referenceImageCount = referenceImages.length;

  const appendReferenceImages = useCallback(
    (images: GenerateReferenceImageAsset[]) => {
      onCommit((currentSettings) => {
        const remaining = getMaxReferenceImages(selectedModel) - currentSettings.referenceImages.length;

        if (remaining <= 0 || images.length === 0) {
          return currentSettings;
        }

        return {
          ...currentSettings,
          referenceImages: [
            ...currentSettings.referenceImages,
            ...images.slice(0, remaining).map((image) => ({
              config: getDefaultReferenceImageConfig(selectedModel, models, image),
              id: createReferenceImageId(),
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

  const clearReferenceImages = useCallback(() => onCommitImmediate({ referenceImages: [] }), [onCommitImmediate]);

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

  const uploadFiles = useCallback(
    async (files: File[]) => {
      if (!canAdd || files.length === 0) {
        return;
      }

      try {
        const uploaded = await Promise.all(
          files.slice(0, maxReferenceImages - referenceImageCount).map((file) => galleryTransfers.upload(file, 'none'))
        );

        appendReferenceImages(uploaded.map(generatedImageToReferenceImage));
        gallery.touchImages();
      } catch (error) {
        notifications.reportError({
          area: 'reference-images',
          message: error instanceof Error ? error.message : String(error),
          namespace: 'generation',
        });
      }
    },
    [appendReferenceImages, canAdd, gallery, maxReferenceImages, notifications, referenceImageCount]
  );

  const handleUploadZoneClick = useCallback(() => {
    if (canAdd) {
      fileInputRef.current?.click();
    }
  }, [canAdd]);

  const handleFileInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      void uploadFiles(Array.from(event.currentTarget.files ?? []));
      event.currentTarget.value = '';
    },
    [uploadFiles]
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
        void addGalleryImages(data.images.map((image) => image.imageName));
      }
    },
  });

  const badges = useMemo(() => {
    if (!isSupported) {
      return referenceImageCount > 0 ? (
        <Badge colorPalette="orange" size="xs" variant="surface">
          {t('widgets.generate.incompatible')}
        </Badge>
      ) : null;
    }

    if (activeCount > 0) {
      return (
        <Badge size="xs" variant="surface">
          {t('widgets.generate.activeCount', { count: activeCount })}
        </Badge>
      );
    }

    return referenceImageCount > 0 ? (
      <Badge size="xs" variant="surface">
        {t('widgets.generate.offCount', { count: referenceImageCount })}
      </Badge>
    ) : null;
  }, [activeCount, isSupported, referenceImageCount, t]);

  if (!isSupported && referenceImageCount === 0) {
    return null;
  }

  // Leftover reference images on a model that cannot use them (e.g. a persisted
  // project). No editable cards, no add paths — just the way out.
  if (!isSupported) {
    return (
      <GenerateCollapsibleSection
        label={t('widgets.generate.referenceImages')}
        defaultOpen
        badges={badges}
        sectionId="reference-images"
      >
        <HStack gap="2" justify="space-between" p="2">
          <Text color="fg.muted" fontSize="2xs" minW="0">
            {t('widgets.generate.referenceImagesUnsupported')}
          </Text>
          <Button colorPalette="red" flexShrink="0" size="xs" variant="outline" onClick={clearReferenceImages}>
            {t('widgets.generate.clearReferenceImages')}
          </Button>
        </HStack>
      </GenerateCollapsibleSection>
    );
  }

  return (
    <GenerateCollapsibleSection
      label={t('widgets.generate.referenceImages')}
      defaultOpen
      badges={badges}
      sectionId="reference-images"
    >
      <Stack ref={setNodeRef} gap="2" p="2">
        <HStack align="stretch" gap="2">
          <DropZone
            as="button"
            alignItems="center"
            cursor={canAdd ? 'pointer' : 'not-allowed'}
            display="flex"
            flex="1"
            fontSize="2xs"
            gap="2"
            isOver={isOver}
            justifyContent="center"
            minH="12"
            minW="0"
            opacity={canAdd ? 1 : 0.6}
            px="3"
            _hover={canAdd ? UPLOAD_ZONE_HOVER_STYLES : undefined}
            onClick={handleUploadZoneClick}
          >
            <UploadIcon size="14" />
            {t('widgets.generate.referenceImagesHelp')}
          </DropZone>
        </HStack>

        {referenceImageCount > 0 ? (
          <Stack gap="2">
            {referenceImages.map((referenceImage, index) => (
              <ReferenceImageCard
                key={referenceImage.id}
                index={index}
                referenceImage={referenceImage}
                selectedModel={selectedModel}
                onPatch={patchReferenceImage}
                onRemove={removeReferenceImage}
                onUseSize={applyReferenceImageSize}
              />
            ))}
          </Stack>
        ) : null}

        <Input
          ref={fileInputRef}
          accept="image/*"
          display="none"
          multiple
          type="file"
          onChange={handleFileInputChange}
        />
      </Stack>
    </GenerateCollapsibleSection>
  );
};

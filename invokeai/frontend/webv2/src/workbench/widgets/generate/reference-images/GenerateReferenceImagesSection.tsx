import type {
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateSettings,
} from '@workbench/generation/types';
import type { ModelConfig } from '@workbench/models/types';
import type { GenerateSettingsUpdate } from '@workbench/widgets/generate/generateDebounce';
import type { ChangeEvent } from 'react';

import { Badge, Box, HStack, Input, Stack, Text } from '@chakra-ui/react';
import { useDndMonitor, useDroppable } from '@dnd-kit/core';
import { Button, IconButton, Tooltip } from '@workbench/components/ui';
import { getGalleryImagesByNames, uploadGalleryImage } from '@workbench/gallery/api';
import {
  createReferenceImageId,
  getDefaultReferenceImageConfig,
  getGenerationDimensions,
  getMaxReferenceImages,
  isReferenceImageSupported,
} from '@workbench/generation/baseGenerationPolicies';
import { generatedImageToReferenceImage, getEffectiveReferenceImage } from '@workbench/generation/referenceImage';
import { clampDimension, deriveAspectRatioId } from '@workbench/generation/settings';
import { isGalleryImageDragData } from '@workbench/widgets/gallery/galleryDnd';
import { GenerateCollapsibleSection } from '@workbench/widgets/generate/shared/GenerateCollapsibleSection';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { ScanIcon, UploadIcon } from 'lucide-react';
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
  const dispatch = useWorkbenchDispatch();
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
          files.slice(0, maxReferenceImages - referenceImageCount).map((file) => uploadGalleryImage(file, 'none'))
        );

        appendReferenceImages(uploaded.map(generatedImageToReferenceImage));
        dispatch({ type: 'touchGalleryImagesRefresh' });
      } catch (error) {
        dispatch({
          area: 'reference-images',
          message: error instanceof Error ? error.message : String(error),
          namespace: 'generation',
          type: 'recordError',
        });
      }
    },
    [appendReferenceImages, canAdd, dispatch, maxReferenceImages, referenceImageCount]
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

    const images = await getGalleryImagesByNames(imageNames.slice(0, maxReferenceImages - referenceImageCount));

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
        <Badge colorPalette="green" size="xs" variant="surface">
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
      <GenerateCollapsibleSection label={t('widgets.generate.referenceImages')} defaultOpen badges={badges}>
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
    <GenerateCollapsibleSection label={t('widgets.generate.referenceImages')} defaultOpen badges={badges}>
      <Stack ref={setNodeRef} bg={isOver ? 'accent.muted' : undefined} gap="2" p="2">
        <HStack align="stretch" gap="2">
          <Box
            as="button"
            alignItems="center"
            borderColor={isOver ? 'accent.solid' : 'border.emphasized'}
            borderStyle="dashed"
            borderWidth="1px"
            color="fg.muted"
            cursor={canAdd ? 'pointer' : 'not-allowed'}
            display="flex"
            flex="1"
            fontSize="2xs"
            gap="2"
            justifyContent="center"
            minH="12"
            minW="0"
            opacity={canAdd ? 1 : 0.6}
            px="3"
            rounded="md"
            _hover={canAdd ? UPLOAD_ZONE_HOVER_STYLES : undefined}
            onClick={handleUploadZoneClick}
          >
            <UploadIcon size="14" />
            {t('widgets.generate.referenceImagesHelp')}
          </Box>
          {/* The button is disabled (canvas bbox pulls are not wired up yet), so
              the tooltip hangs off a wrapper — disabled buttons eat pointer events. */}
          <Tooltip content={t('widgets.generate.pullBboxIntoReferenceImage')}>
            <Box display="flex" flexShrink="0">
              <IconButton
                aria-label={t('widgets.generate.pullBboxIntoReferenceImage')}
                disabled
                h="full"
                minH="12"
                minW="12"
                variant="outline"
              >
                <ScanIcon />
              </IconButton>
            </Box>
          </Tooltip>
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

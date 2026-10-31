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
import type { MouseEvent } from 'react';

import { HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { useDndMonitor } from '@dnd-kit/core';
import { galleryImages, toGalleryItemKey } from '@features/gallery';
import { GalleryPickerPopover } from '@features/gallery/picker';
import { isGalleryImageDragData, useGalleryImageDroppable } from '@features/gallery/utility';
import {
  createReferenceImageId,
  getGenerationDimensions,
  getMaxReferenceImages,
  getMissingReferenceImageFormat,
  isReferenceImageSupported,
  withAddedReferenceImages,
} from '@features/generation/core/baseGenerationPolicies';
import { generatedImageToReferenceImage, getEffectiveReferenceImage } from '@features/generation/core/referenceImage';
import {
  clampDimension,
  deriveAspectRatioId,
  moveReferenceImage,
  normalizeGenerateWidgetValues,
} from '@features/generation/core/settings';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { GenerateIssueNotes } from '@features/generation/ui/shared/GenerateIssues';
import { Button, DropTargetOverlay, DropZone } from '@platform/ui';
import { ChevronDownIcon, ImagePlusIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { ReferenceImageCard } from './ReferenceImageCard';

const UPLOAD_ZONE_HOVER_STYLES = { bg: 'bg.muted', color: 'fg' };
/** Display names for the formats the backend can require of reference-image models. */
const MODEL_FORMAT_LABELS: Record<string, string> = { diffusers: 'Diffusers' };
const IMAGE_ONLY = ['image'] as const;
/** The picker's trigger skips a click whose default is prevented, so a full set opens no picker. */
const preventPickerOpen = (event: MouseEvent) => event.preventDefault();

interface GenerateReferenceImagesContentProps {
  models: readonly ModelConfig[];
  referenceImages: GenerateReferenceImage[];
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (update: GenerateSettingsUpdate) => void;
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

export const GenerateReferenceImagesContent = ({
  models,
  onCommit,
  onCommitImmediate,
  referenceImages,
  selectedModel,
}: GenerateReferenceImagesContentProps) => {
  const { t } = useTranslation();
  const { gallery, models: modelUi, notifications, project, settings } = useGenerationUi();
  const isSupported = isReferenceImageSupported(selectedModel);
  const missingFormat = getMissingReferenceImageFormat(selectedModel);
  const maxReferenceImages = getMaxReferenceImages(selectedModel);
  const referenceImageCount = referenceImages.length;
  const freeSlots = Math.max(0, maxReferenceImages - referenceImageCount);
  const canAdd = freeSlots > 0;
  // Still a target when full, so a drop there says why nothing was added instead of doing nothing.
  const { acceptsActiveDrag, isOver, setNodeRef } = useGalleryImageDroppable({
    data: { kind: 'generate-reference-images' },
    disabled: !isSupported,
    id: 'generate-reference-images',
  });

  /** Adds what fits and returns how many that was, as the commit counted the free slots. */
  const appendReferenceImages = useCallback(
    (images: GenerateReferenceImageAsset[]): number => {
      // Mint IDs outside replayed updaters so draft and flushed identities match.
      const ids = images.map(() => createReferenceImageId());
      let added = 0;

      // The commit applies the update at once (a later flush may replay it), so `added` is set when this returns.
      onCommit((currentSettings) => {
        const remaining = getMaxReferenceImages(selectedModel) - currentSettings.referenceImages.length;

        added = Math.max(0, Math.min(remaining, images.length));

        if (added === 0) {
          return currentSettings;
        }

        return {
          ...currentSettings,
          referenceImages: withAddedReferenceImages(
            selectedModel,
            models,
            currentSettings.referenceImages,
            images.slice(0, remaining).map((image, index) => ({ id: ids[index] ?? createReferenceImageId(), image }))
          ),
        };
      });

      return added;
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

  // Undoable, so it does not ask first. The Undo puts the removed references back ahead of any added since.
  const clearReferenceImages = useCallback(() => {
    const { activeProjectId: projectId } = project;
    const removed = referenceImages;

    onCommitImmediate({ referenceImages: [] });
    notifications.undoable(t('widgets.generate.referenceImagesRemoved', { count: removed.length }), {
      onUndo: () => {
        const values = normalizeGenerateWidgetValues(settings.readGenerateValues(projectId));
        const current = values?.referenceImages ?? [];
        const removedIds = new Set(removed.map((referenceImage) => referenceImage.id));
        const restored = [...removed, ...current.filter((referenceImage) => !removedIds.has(referenceImage.id))];
        // A model that takes references gets no more running than it takes: the rest come back turned off.
        const limit = values && isReferenceImageSupported(values.model) ? getMaxReferenceImages(values.model) : null;
        let running = 0;

        settings.patchGenerateSettings(
          {
            referenceImages:
              limit === null
                ? restored
                : restored.map((referenceImage) => {
                    if (!referenceImage.isEnabled) {
                      return referenceImage;
                    }

                    running += 1;
                    return running > limit ? { ...referenceImage, isEnabled: false } : referenceImage;
                  }),
          },
          projectId
        );
      },
      projectId,
    });
  }, [notifications, onCommitImmediate, project, referenceImages, settings, t]);

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

  // A drop never loses images silently: what did not fit is reported with the limit.
  const addGalleryImages = async (imageNames: string[]) => {
    if (imageNames.length === 0) {
      return;
    }

    if (!canAdd) {
      notifications.info(
        t('widgets.generate.referenceSlotsFull', { count: maxReferenceImages }),
        t('widgets.generate.referenceSlotsFullHint')
      );
      return;
    }

    const images = await galleryImages.resolveMany(imageNames.slice(0, freeSlots));
    const added = images.length > 0 ? appendReferenceImages(images.map(generatedImageToReferenceImage)) : 0;

    if (imageNames.length > freeSlots) {
      notifications.info(
        t('widgets.generate.referenceImagesPartlyAdded', { added, count: imageNames.length }),
        t('widgets.generate.referenceSlotsLimit', { count: maxReferenceImages })
      );
    }
  };

  useDndMonitor({
    onDragEnd: (event) => {
      const data = event.active.data.current;

      if (event.over?.id === 'generate-reference-images' && isGalleryImageDragData(data)) {
        void addGalleryImages(data.items.map((item) => item.name));
      }
    },
  });

  // A model that only lacks the right format says which one would take references.
  const unsupportedMessage =
    selectedModel && missingFormat
      ? t('widgets.generate.referenceImagesNeedFormat', {
          base: modelUi.getBaseLabel(selectedModel.base),
          format: MODEL_FORMAT_LABELS[missingFormat] ?? missingFormat,
        })
      : t('widgets.generate.referenceImagesUnsupported');

  if (!isSupported && referenceImageCount === 0) {
    return missingFormat ? (
      <Text color="fg.muted" fontSize="xs">
        {unsupportedMessage}
      </Text>
    ) : null;
  }

  // Unsupported-model references remain read-only with removal as recovery.
  if (!isSupported) {
    return (
      <Stack gap="2">
        <HStack gap="2" justify="space-between">
          <Text color="fg.muted" fontSize="xs" minW="0">
            {unsupportedMessage}
          </Text>
          <Button colorPalette="red" flexShrink="0" variant="outline" onClick={clearReferenceImages}>
            {t('widgets.generate.clearReferenceImages')}
          </Button>
        </HStack>
        {/* No cards to show them on: every reference's issue lists here. */}
        <GenerateIssueNotes field="referenceImage" section="guidance" />
      </Stack>
    );
  }

  return (
    <Stack ref={setNodeRef} gap="2" position="relative">
      <DropTargetOverlay
        isActive={acceptsActiveDrag}
        isOver={isOver}
        label={
          canAdd
            ? t('widgets.generate.dropReferenceImages', { count: freeSlots })
            : t('widgets.generate.referenceSlotsFull', { count: maxReferenceImages })
        }
      />
      <GalleryPickerPopover
        accept={IMAGE_ONLY}
        label={t('widgets.generate.addReferenceImage')}
        selection={pickerSelection}
        onPick={handlePick}
      >
        {/* Full, it stays a focusable drop target that says why it adds nothing: `aria-disabled` rather than `disabled`,
            since Chrome sends no mouseup to a disabled control and a drag released over one would never end. */}
        <DropZone
          as="button"
          alignItems="center"
          aria-disabled={canAdd ? undefined : true}
          cursor={canAdd ? undefined : 'not-allowed'}
          display="flex"
          flexDirection="column"
          fontSize="xs"
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
          onClickCapture={canAdd ? undefined : preventPickerOpen}
        >
          <HStack as="span" color="fg" fontSize="md" fontWeight="600" gap="1.5">
            <Icon as={ImagePlusIcon} boxSize="4" />
            {t('widgets.generate.addReferenceImage')}
            <Icon as={ChevronDownIcon} boxSize="3" color="fg.subtle" />
          </HStack>
          <Text as="span" color="fg.muted" textAlign="center" textWrap="pretty">
            {canAdd
              ? t('widgets.generate.referenceSlotsHint', { count: referenceImageCount, max: maxReferenceImages })
              : t('widgets.generate.referenceSlotsFullZone', { count: maxReferenceImages })}
          </Text>
        </DropZone>
      </GalleryPickerPopover>

      {/* Issues about the references as a group (their count); each card shows its own. */}
      <GenerateIssueNotes field="referenceImages" section="guidance" />

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

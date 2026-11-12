import type { GalleryItem } from '@features/gallery';
import type { GalleryPickerSelection } from '@features/gallery/picker';
import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { ReferenceVisionComponents } from '@features/generation/core/baseGenerationPolicies';
import type {
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateSettings,
} from '@features/generation/core/types';
import type { GenerateSettingsUpdate } from '@features/generation/ui/generateDebounce';
import type { MouseEvent } from 'react';

import { Box, chakra, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { useDndMonitor } from '@dnd-kit/core';
import { galleryImages, toGalleryItemKey } from '@features/gallery';
import { GalleryPickerPopover } from '@features/gallery/picker';
import { isGalleryImageDragData, useGalleryImageDroppable } from '@features/gallery/utility';
import {
  createReferenceImageId,
  getGenerationDimensions,
  getMaxReferenceImages,
  getReferenceImageUnavailableReason,
  isReferenceImageSupported,
  withAddedReferenceImages,
} from '@features/generation/core/baseGenerationPolicies';
import {
  generatedImageToReferenceImage,
  getEffectiveReferenceImage,
  getReferenceImagePromptTags,
} from '@features/generation/core/referenceImage';
import {
  clampDimension,
  deriveAspectRatioId,
  moveReferenceImage,
  normalizeGenerateWidgetValues,
  renameReferenceImage,
} from '@features/generation/core/settings';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { GenerateIssueNotes } from '@features/generation/ui/shared/GenerateIssues';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { Button } from '@platform/ui/Button';
import { DropTargetOverlay } from '@platform/ui/DropTargetOverlay';
import { FieldLabel } from '@platform/ui/Field';
import { FeatureHint } from '@platform/ui/hints';
import { ListStack } from '@platform/ui/list/ListStack';
import { Row } from '@platform/ui/Row';
import { PlusIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { getReferenceModelTypes, ReferenceImageRow } from './ReferenceImageRow';

const IMAGE_ONLY = ['image'] as const;
const FOCUS_WAIT_FRAMES = 30;

/** An undone removal brings its row back with the settings store's next render; it takes focus once it is there. */
const focusReferenceWhenRendered = (id: string, framesLeft = FOCUS_WAIT_FRAMES) =>
  requestAnimationFrame(() => {
    const row = document.querySelector<HTMLElement>(
      `[data-reference-image-id="${CSS.escape(id)}"] [data-list-primary]`
    );

    if (row) {
      row.focus();
    } else if (framesLeft > 0) {
      focusReferenceWhenRendered(id, framesLeft - 1);
    }
  });
/** The picker's trigger skips a click whose default is prevented, so a full set opens no picker. */
const preventPickerOpen = (event: MouseEvent) => event.preventDefault();

interface GenerateReferenceImagesContentProps {
  models: readonly ModelConfig[];
  referenceImages: GenerateReferenceImage[];
  selectedModel: GenerateModelConfig | undefined;
  /** The selected Qwen3-VL encoder and vision tower: for some bases they decide whether the references can be read. */
  visionComponents: ReferenceVisionComponents;
  onCommit: (update: GenerateSettingsUpdate) => void;
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

export const GenerateReferenceImagesContent = ({
  models,
  onCommit,
  onCommitImmediate,
  referenceImages,
  selectedModel,
  visionComponents,
}: GenerateReferenceImagesContentProps) => {
  const { t } = useTranslation();
  const { gallery, models: uiModels, notifications, project, settings } = useGenerationUi();
  const { openManager } = uiModels;
  const openModelManager = useCallback((modelType: string) => openManager({ modelType }), [openManager]);
  const isSupported = isReferenceImageSupported(selectedModel, visionComponents);
  const unavailableReason = getReferenceImageUnavailableReason(selectedModel, visionComponents);
  const maxReferenceImages = getMaxReferenceImages(selectedModel, visionComponents);
  const referenceImageCount = referenceImages.length;
  const freeSlots = Math.max(0, maxReferenceImages - referenceImageCount);
  const canAdd = freeSlots > 0;
  // Still a target when full, so a drop there says why nothing was added instead of doing nothing.
  const { acceptsActiveDrag, isOver, setNodeRef } = useGalleryImageDroppable({
    data: { kind: 'generate-reference-images' },
    disabled: !isSupported,
    id: 'generate-reference-images',
  });
  const promptTags = useMemo(
    () => getReferenceImagePromptTags(selectedModel?.base, referenceImages),
    [referenceImages, selectedModel?.base]
  );

  /** Adds what fits and returns how many that was, as the commit counted the free slots. */
  const appendReferenceImages = useCallback(
    (images: GenerateReferenceImageAsset[]): number => {
      // Mint IDs outside replayed updaters so draft and flushed identities match.
      const ids = images.map(() => createReferenceImageId());
      let added = 0;

      // The commit applies the update at once (a later flush may replay it), so `added` is set when this returns.
      onCommit((currentSettings) => {
        const remaining =
          getMaxReferenceImages(selectedModel, currentSettings) - currentSettings.referenceImages.length;

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

  const handleRenameReferenceImage = useCallback(
    (id: string, name: string, isDefaultName: (name: string) => boolean) => {
      onCommit((currentSettings) => {
        const referenceImages = renameReferenceImage(currentSettings.referenceImages, id, name, isDefaultName);

        return referenceImages === currentSettings.referenceImages
          ? currentSettings
          : { ...currentSettings, referenceImages: [...referenceImages] };
      });
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

  /**
   * Puts removed references back at `at` among those there now (any added since stay), each as it was: name, crop and
   * settings. A model that takes references gets no more running than it takes: the rest come back turned off.
   */
  const restoreReferenceImages = useCallback(
    (removed: readonly GenerateReferenceImage[], at: number, projectId: string) => {
      const values = normalizeGenerateWidgetValues(settings.readGenerateValues(projectId));
      const removedIds = new Set(removed.map((referenceImage) => referenceImage.id));
      const kept = (values?.referenceImages ?? []).filter((referenceImage) => !removedIds.has(referenceImage.id));
      const position = Math.min(at, kept.length);
      const restored = [...kept.slice(0, position), ...removed, ...kept.slice(position)];
      const limit =
        values && isReferenceImageSupported(values.model, values) ? getMaxReferenceImages(values.model, values) : null;
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
    [settings]
  );

  // Undoable, so it does not ask first: the Undo puts the reference back where it was, as it was.
  const removeReferenceImage = useCallback(
    (id: string) => {
      const { activeProjectId: projectId } = project;
      let found: { at: number; referenceImage: GenerateReferenceImage } | null = null;

      // The commit applies the update at once, so `found` is set when it returns; a later replay finds the same.
      onCommit((currentSettings) => {
        const at = currentSettings.referenceImages.findIndex((referenceImage) => referenceImage.id === id);
        const referenceImage = currentSettings.referenceImages[at];

        if (!referenceImage) {
          return currentSettings;
        }

        found ??= { at, referenceImage };
        return {
          ...currentSettings,
          referenceImages: currentSettings.referenceImages.filter((candidate) => candidate.id !== id),
        };
      });

      const removed = found as { at: number; referenceImage: GenerateReferenceImage } | null;

      if (!removed) {
        return;
      }

      notifications.undoable(
        t('widgets.generate.referenceImageRemoved', {
          name: removed.referenceImage.name ?? t('widgets.generate.referenceImageNumbered', { index: removed.at + 1 }),
        }),
        {
          onUndo: () => {
            // The removal may still sit in the form's debounce; it lands before the restore reads the settings.
            flushWorkbenchDrafts();
            restoreReferenceImages([removed.referenceImage], removed.at, projectId);
            focusReferenceWhenRendered(id);
          },
          projectId,
        }
      );
    },
    [notifications, onCommit, project, restoreReferenceImages, t]
  );

  // Undoable, so it does not ask first. The Undo puts the removed references back ahead of any added since.
  const clearReferenceImages = useCallback(() => {
    const { activeProjectId: projectId } = project;
    const removed = referenceImages;

    onCommitImmediate({ referenceImages: [] });
    notifications.undoable(t('widgets.generate.referenceImagesRemoved', { count: removed.length }), {
      onUndo: () => restoreReferenceImages(removed, 0, projectId),
      projectId,
    });
  }, [notifications, onCommitImmediate, project, referenceImages, restoreReferenceImages, t]);

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

  // A model whose encoder lacks the vision tower says what would take references.
  const unsupportedMessage =
    unavailableReason === 'ggufVisionTower'
      ? t('widgets.generate.referenceImagesNeedGgufVisionTower')
      : unavailableReason
        ? t('widgets.generate.referenceImagesNeedVisionTower')
        : t('widgets.generate.referenceImagesUnsupported');

  if (!isSupported && referenceImageCount === 0) {
    return unavailableReason ? (
      <Text color="fg.muted" fontSize="xs">
        {unsupportedMessage}
      </Text>
    ) : null;
  }

  const referenceModelTypes = getReferenceModelTypes(selectedModel?.base);
  const hasModelChoice = models.some(
    (model) => model.base === selectedModel?.base && referenceModelTypes.includes(model.type)
  );
  const rows = referenceImages.map((referenceImage, index) => (
    <ReferenceImageRow
      key={referenceImage.id}
      count={referenceImageCount}
      hasModelChoice={hasModelChoice}
      onOpenModelManager={openModelManager}
      index={index}
      isSupported={isSupported}
      promptTag={promptTags.get(referenceImage.id)}
      referenceImage={referenceImage}
      selectedModel={selectedModel}
      onFindInGallery={gallery.findImage}
      onMove={handleMoveReferenceImage}
      onPatch={patchReferenceImage}
      onRemove={removeReferenceImage}
      onRename={handleRenameReferenceImage}
      onUseSize={applyReferenceImageSize}
    />
  ));
  const label = (
    <Box>
      <FeatureHint hint="referenceImage">
        <chakra.span>
          <FieldLabel>{t('widgets.generate.referenceImages')}</FieldLabel>
        </chakra.span>
      </FeatureHint>
    </Box>
  );

  // References the model cannot use stay listed, inert, so it is clear what Remove all would clear.
  if (!isSupported) {
    return (
      <Stack gap="1.5">
        {label}
        <HStack gap="2" justify="space-between">
          <Text color="fg.muted" fontSize="xs" minW="0">
            {unsupportedMessage}
          </Text>
          <Button colorPalette="red" flexShrink="0" variant="outline" onClick={clearReferenceImages}>
            {t('widgets.generate.clearReferenceImages')}
          </Button>
        </HStack>
        {/* The rows are inert here, so every reference's issue lists above them, with its fix. */}
        <GenerateIssueNotes field="referenceImage" section="guidance" />
        <Box mx={-1}>
          <ListStack dividers label={t('widgets.generate.referenceImages')}>
            {rows}
          </ListStack>
        </Box>
      </Stack>
    );
  }

  return (
    <Stack ref={setNodeRef} gap="1.5" position="relative">
      <DropTargetOverlay
        isActive={acceptsActiveDrag}
        isOver={isOver}
        label={
          canAdd
            ? t('widgets.generate.dropReferenceImages', { count: freeSlots })
            : t('widgets.generate.referenceSlotsFull', { count: maxReferenceImages })
        }
      />
      {label}
      {/* Issues about the references as a group (their count); each row shows its own. */}
      <GenerateIssueNotes field="referenceImages" section="guidance" />
      {/* The same bleed as the concept list below it, so both lists' rows share their edges and inner padding. */}
      <Box mx={-1}>
        <ListStack dividers label={t('widgets.generate.referenceImages')}>
          {rows}
          {/* The list's last row, and its whole empty state. */}
          <Box role="listitem">
            <GalleryPickerPopover
              accept={IMAGE_ONLY}
              label={t('widgets.generate.addReferenceImage')}
              selection={pickerSelection}
              onPick={handlePick}
            >
              {/* Full, it stays a focusable drop target that says why it adds nothing: `aria-disabled` rather than
                  `disabled`, since Chrome sends no mouseup to a disabled control and a drag released over one would
                  never end. */}
              <Row
                asChild
                aria-disabled={canAdd ? undefined : true}
                cursor={canAdd ? undefined : 'not-allowed'}
                data-disabled={canAdd ? undefined : ''}
                data-hover={isOver && canAdd ? '' : undefined}
                data-list-surface=""
                data-reference-add=""
                gap="2.5"
                minH="12"
                px="1.5"
                py="1.5"
                onClickCapture={canAdd ? undefined : preventPickerOpen}
              >
                <chakra.button type="button">
                  <Box
                    alignItems="center"
                    borderColor="border.emphasized"
                    borderStyle="dashed"
                    borderWidth="1px"
                    boxSize="9"
                    color="fg.muted"
                    display="flex"
                    flexShrink={0}
                    justifyContent="center"
                    rounded="sm"
                  >
                    <Icon as={PlusIcon} boxSize="3.5" />
                  </Box>
                  <Stack as="span" flex="1" gap="0.5" minW="0">
                    <Text as="span" fontSize="md" fontWeight="600" lineHeight="shorter" truncate>
                      {t('widgets.generate.addReferenceImage')}
                    </Text>
                    <Text as="span" color="fg.muted" fontSize="xs" lineHeight="shorter" textWrap="pretty">
                      {canAdd
                        ? t('widgets.generate.referenceSlotsHint', {
                            count: referenceImageCount,
                            max: maxReferenceImages,
                          })
                        : t('widgets.generate.referenceSlotsFullZone', { count: maxReferenceImages })}
                    </Text>
                  </Stack>
                </chakra.button>
              </Row>
            </GalleryPickerPopover>
          </Box>
        </ListStack>
      </Box>
    </Stack>
  );
};

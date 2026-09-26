import type { DragEndEvent } from '@dnd-kit/core';
import type { GalleryItem, GalleryVideoItem } from '@features/gallery';
import type { GalleryPickerSelection } from '@features/gallery/picker';
import type {
  VideoReferenceConditioning,
  VideoReferenceImageDetail,
  VideoReferenceItem,
} from '@features/video/core/types';
import type { ChangeEvent } from 'react';

import { Badge, Box, createListCollection, HStack, Icon, Image, Input, Spinner, Stack, Text } from '@chakra-ui/react';
import { useDndContext, useDndMonitor, useDroppable } from '@dnd-kit/core';
import { galleryItems, galleryTransfers, toGalleryItemKey } from '@features/gallery';
import { FindInGalleryThumbnailButton } from '@features/gallery/mediaSlot';
import { getGalleryUploadAccept, GalleryPickerPopover } from '@features/gallery/picker';
import { galleryImageUrls, galleryVideoUrls, isGalleryItemDragData } from '@features/gallery/utility';
import { resolveMiniMaxH3ReferenceImage } from '@features/video/core/dimensions';
import {
  clampReferenceSampleFrames,
  createVideoReferenceEntry,
  formatReferencePromptLabels,
  getDefaultReferenceImageDetail,
  referencePromptLabels,
  referenceSampleFrames,
  resizeReferenceSampleWindow,
  slideReferenceSampleWindow,
} from '@features/video/core/settings';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { Button, IconButton } from '@platform/ui/Button';
import { DropTargetOverlay } from '@platform/ui/DropTargetOverlay';
import { DropZone } from '@platform/ui/DropZone';
import { Field } from '@platform/ui/Field';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { Select } from '@platform/ui/Select';
import { ArrowDownIcon, ArrowUpIcon, ChevronDownIcon, ImagePlusIcon, UploadIcon, XIcon } from 'lucide-react';
import { memo, useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { PlayClipSpanButton } from './PlayClipSpanButton';
import { TrimBoundThumb } from './TrimBoundThumb';
import { useVideoUiActions } from './VideoUiContext';

/** Reference order determines prompt labels and generation semantics. */

const DROP_ID = 'video-reference-list';
const IMAGE_UPLOAD_ACCEPT = getGalleryUploadAccept(['image']);
// Audio uploads become waveform videos and consume the shared video-reference cap.
const MEDIA_UPLOAD_ACCEPT = getGalleryUploadAccept(['video']);
const DROP_ZONE_FOCUS_PROPS = {
  outlineColor: 'accent.focusRing',
  outlineOffset: '2px',
  outlineStyle: 'solid',
  outlineWidth: '2px',
};
const DROP_ZONE_DISABLED_PROPS = { cursor: 'not-allowed', opacity: 0.6 };
const DROP_ZONE_BUSY_PROPS = { disabled: true };
const ACCEPT_MEDIA = ['image', 'video'] as const;

const getSingleGalleryDragItem = (data: unknown): { kind: 'image' | 'video'; name: string } | null => {
  if (!isGalleryItemDragData(data) || data.items.length !== 1) {
    return null;
  }

  const item = data.items[0];

  return item && (item.kind === 'image' || item.kind === 'video') ? { kind: item.kind, name: item.name } : null;
};

type ReferenceCollections = {
  /** The anchor's options: 'Audio only' is absent, see `anchorReferenceConditioning`. */
  anchorConditioning: ReturnType<typeof createListCollection<{ label: string; value: string }>>;
  conditioning: ReturnType<typeof createListCollection<{ label: string; value: string }>>;
  detail: ReturnType<typeof createListCollection<{ label: string; value: string }>>;
};

const ReferenceCard = memo(function ReferenceCard({
  audioLabel,
  collections,
  disabled,
  index,
  canMoveDown,
  canMoveUp,
  onMove,
  onRemove,
  onUpdate,
  pictureLabel,
  reference,
  targetArea,
  videoLabel,
}: {
  /**
   * Pass scalar labels: labels depend on the whole list, but fresh per-card objects would rerender every card on
   * each trim step.
   */
  audioLabel: number | null;
  collections: ReferenceCollections;
  disabled: boolean;
  index: number;
  canMoveDown: boolean;
  canMoveUp: boolean;
  pictureLabel: number | null;
  videoLabel: number | null;
  onMove: (index: number, direction: -1 | 1) => void;
  onRemove: (index: number) => void;
  onUpdate: (index: number, reference: VideoReferenceItem) => void;
  reference: VideoReferenceItem;
  /** The generation's pixel area, which is what 'match' detail scales an image to. */
  targetArea: number | null;
}) {
  const { t } = useTranslation();
  const { findInGallery } = useVideoUiActions();
  const moveUpRef = useRef<HTMLButtonElement>(null);
  const moveDownRef = useRef<HTMLButtonElement>(null);
  const name = reference.kind === 'video' ? reference.clip.video_name : reference.image.image_name;
  const kind = reference.kind;
  // Finding media stays available while editing is disabled; expose one action per media record.
  const findReferenceInGallery = useCallback(() => findInGallery({ kind, name }), [findInGallery, kind, name]);
  const promptLabels = useMemo(
    () => formatReferencePromptLabels({ audio: audioLabel, picture: pictureLabel, video: videoLabel }),
    [audioLabel, pictureLabel, videoLabel]
  );
  const selectValue = useMemo(
    () => [reference.kind === 'video' ? reference.conditioning : reference.detail],
    [reference]
  );
  const selectCollection = useMemo(() => {
    if (reference.kind !== 'video') {
      return collections.detail;
    }

    // The extension anchor needs visual rows, so Audio only is unavailable.
    return reference.fromSourceVideo === true ? collections.anchorConditioning : collections.conditioning;
  }, [collections, reference]);
  const handleSelect = useCallback(
    (details: { value: string[] }) => {
      const value = details.value[0];

      if (!value) {
        return;
      }
      if (reference.kind === 'video') {
        onUpdate(index, { ...reference, conditioning: value as VideoReferenceConditioning });
      } else {
        onUpdate(index, { ...reference, detail: value as VideoReferenceImageDetail });
      }
    },
    [index, onUpdate, reference]
  );
  // Present start plus length while storing request bounds. Editing either control marks the anchor window
  // overridden; derivation from the initial-video cutpoint is only a default.
  const handleStartFrame = useCallback(
    (rawStart: number) => {
      if (reference.kind === 'video') {
        // Capture pre-drag length once so subsequent steps slide the same window reversibly.
        const sampleFrames = referenceSampleFrames(reference);

        onUpdate(index, {
          ...reference,
          clip: slideReferenceSampleWindow(reference.clip, rawStart, sampleFrames),
          sampleFrames,
          ...(reference.fromSourceVideo === true ? { trimOverridden: true } : {}),
        });
      }
    },
    [index, onUpdate, reference]
  );
  const handleSampleFrames = useCallback(
    (rawSampleFrames: number) => {
      if (reference.kind === 'video') {
        onUpdate(index, {
          ...reference,
          clip: resizeReferenceSampleWindow(reference.clip, rawSampleFrames),
          sampleFrames: clampReferenceSampleFrames(reference.clip, rawSampleFrames),
          ...(reference.fromSourceVideo === true ? { trimOverridden: true } : {}),
        });
      }
    },
    [index, onUpdate, reference]
  );
  const imageCost = useMemo(
    () =>
      reference.kind === 'image'
        ? resolveMiniMaxH3ReferenceImage(reference.image.width, reference.image.height, reference.detail, targetArea)
        : null,
    [reference, targetArea]
  );
  // Store frames for the trim contract; show seconds to make sample duration readable.
  const sampleFrames = reference.kind === 'video' ? reference.clip.endFrame - reference.clip.startFrame + 1 : 0;
  const sampleSeconds =
    reference.kind === 'video' && Number.isFinite(reference.clip.fps) && reference.clip.fps > 0
      ? (sampleFrames / reference.clip.fps).toFixed(1)
      : null;
  // Arm focus repair only if the arrow owns focus. Reordering can disable it; inspect the committed disabled state
  // because the anchor rule determines the endpoint. Some browsers click without focusing, so never steal focus
  // from the prompt.
  const pendingFocusRef = useRef<'down' | 'up' | null>(null);
  const handleMoveUp = useCallback(() => {
    const arrow = moveUpRef.current;

    pendingFocusRef.current = arrow !== null && document.activeElement === arrow ? 'up' : null;
    onMove(index, -1);
  }, [index, onMove]);
  const handleMoveDown = useCallback(() => {
    const arrow = moveDownRef.current;

    pendingFocusRef.current = arrow !== null && document.activeElement === arrow ? 'down' : null;
    onMove(index, 1);
  }, [index, onMove]);

  useLayoutEffect(() => {
    const pending = pendingFocusRef.current;

    if (pending === null) {
      return;
    }
    pendingFocusRef.current = null;

    const pressed = pending === 'up' ? moveUpRef.current : moveDownRef.current;
    const sibling = pending === 'up' ? moveDownRef.current : moveUpRef.current;
    // Repair only arrow/body focus: a dropped write can leave this arm until an unrelated render, after the user
    // has focused elsewhere.
    const isOursToRestore = document.activeElement === document.body || document.activeElement === pressed;

    if (isOursToRestore && pressed?.disabled === true && sibling !== null && !sibling.disabled) {
      sibling.focus();
    }
  });
  const handleRemove = useCallback(() => onRemove(index), [index, onRemove]);

  return (
    /* Prompt labels disambiguate each card's repeated controls. */
    <Box aria-label={[...promptLabels, name].join(' ')} borderWidth="1px" p="2" role="group" rounded="md">
      <HStack align="start" gap="2">
        {reference.kind === 'video' ? <PlayClipSpanButton clip={reference.clip} /> : null}
        {reference.kind === 'image' ? (
          <Box
            bg="blackAlpha.300"
            className="group"
            flexShrink={0}
            h="12"
            overflow="hidden"
            position="relative"
            rounded="sm"
            w="16"
          >
            <Image alt="" fit="cover" h="100%" src={galleryImageUrls.thumbnail(name)} w="100%" />
            <FindInGalleryThumbnailButton name={name} onFind={findReferenceInGallery} />
          </Box>
        ) : null}
        <Stack flex="1" gap="1" minW="0">
          <HStack gap="1">
            {/* Render prompt tokens verbatim in LTR order; modality counters differ from card positions. */}
            {promptLabels.map((label) => (
              <Badge key={label} dir="ltr" flexShrink={0} size="xs" userSelect="text" variant="solid">
                {label}
              </Badge>
            ))}
            <MiddleTruncate flex="1" fontSize="xs" text={name} />
            {reference.kind === 'video' && reference.fromSourceVideo === true ? (
              <Badge flexShrink={0} size="xs" variant="outline">
                {t('widgets.video.referenceFromInitialVideo')}
              </Badge>
            ) : null}
          </HStack>
          <Select
            collection={selectCollection}
            disabled={disabled}
            size="xs"
            value={selectValue}
            onValueChange={handleSelect}
          />
          {imageCost ? (
            <Text color="fg.muted" fontSize="2xs" fontVariantNumeric="tabular-nums">
              {t('widgets.video.referenceImageCost', {
                height: imageCost.dimensions.height,
                rows: imageCost.rows.toLocaleString(),
                width: imageCost.dimensions.width,
              })}
            </Text>
          ) : null}
          {/* The second control edits length; its thumbnail shows the resulting end frame. */}
          {reference.kind === 'video' ? (
            <Stack gap="1">
              <HStack gap="2">
                <TrimBoundThumb
                  fps={reference.clip.fps}
                  frame={reference.clip.startFrame}
                  label={t('widgets.video.trimStartShort')}
                  name={name}
                  src={galleryVideoUrls.full(name)}
                  onFindInGallery={findReferenceInGallery}
                />
                <ScrubberField
                  disabled={disabled}
                  label={t('widgets.video.trimStart')}
                  max={Math.max(0, reference.clip.numFrames - 1)}
                  min={0}
                  step={1}
                  value={reference.clip.startFrame}
                  onChange={handleStartFrame}
                />
              </HStack>
              <HStack gap="2">
                <TrimBoundThumb
                  fps={reference.clip.fps}
                  frame={reference.clip.endFrame}
                  label={`${t('widgets.video.trimEndShort')} · ${reference.clip.endFrame}`}
                  src={galleryVideoUrls.full(name)}
                />
                <ScrubberField
                  disabled={disabled}
                  label={
                    sampleSeconds === null
                      ? t('widgets.video.sampleLength')
                      : t('widgets.video.sampleLengthWithSeconds', { seconds: sampleSeconds })
                  }
                  max={Math.max(1, reference.clip.numFrames - reference.clip.startFrame)}
                  min={1}
                  step={1}
                  value={sampleFrames}
                  onChange={handleSampleFrames}
                />
              </HStack>
            </Stack>
          ) : null}
        </Stack>
        <Stack gap="0">
          <IconButton
            ref={moveUpRef}
            aria-label={t('widgets.video.moveReferenceUp')}
            disabled={disabled || !canMoveUp}
            size="2xs"
            variant="ghost"
            onClick={handleMoveUp}
          >
            <ArrowUpIcon size={12} />
          </IconButton>
          <IconButton
            ref={moveDownRef}
            aria-label={t('widgets.video.moveReferenceDown')}
            disabled={disabled || !canMoveDown}
            size="2xs"
            variant="ghost"
            onClick={handleMoveDown}
          >
            <ArrowDownIcon size={12} />
          </IconButton>
          <IconButton
            aria-label={t('widgets.video.removeReference')}
            disabled={disabled}
            size="2xs"
            variant="ghost"
            onClick={handleRemove}
          >
            <XIcon size={12} />
          </IconButton>
        </Stack>
      </HStack>
    </Box>
  );
});

export const VideoReferenceListField = memo(function VideoReferenceListField({
  disabled = false,
  maxImages,
  maxVideos,
  onChange,
  references,
  targetArea,
}: {
  disabled?: boolean;
  maxImages: number;
  maxVideos: number;
  /** Accept an updater: gallery resolution awaits must not overwrite reference edits made meanwhile. */
  onChange: (update: (current: VideoReferenceItem[]) => VideoReferenceItem[]) => void;
  references: VideoReferenceItem[];
  /** The generation's pixel area, which sizes a 'match'-detail image reference. */
  targetArea: number | null;
}) {
  const { t } = useTranslation();
  const { getUploadBoardId, reportError, touchGalleryImages } = useVideoUiActions();
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const videoInputRef = useRef<HTMLInputElement | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Disable moves that would displace the pinned last anchor and therefore be reverted.
  const anchorIndex = references.findIndex(
    (reference) => reference.kind === 'video' && reference.fromSourceVideo === true
  );
  const videoCount = references.filter((reference) => reference.kind === 'video').length;
  const imageCount = references.length - videoCount;
  const canAddVideo = videoCount < maxVideos;
  const canAddImage = imageCount < maxImages;
  const isInert = disabled || isLoading;

  const { active } = useDndContext();
  const activeDragItem = getSingleGalleryDragItem(active?.data.current);
  const acceptsActiveDrag =
    !isInert && activeDragItem !== null && (activeDragItem.kind === 'video' ? canAddVideo : canAddImage);
  const { isOver, setNodeRef } = useDroppable({ disabled: !acceptsActiveDrag, id: DROP_ID });

  const conditioningItems = useMemo(
    () => [
      { label: t('widgets.video.referenceConditioningVideoAudio'), value: 'video_audio' },
      { label: t('widgets.video.referenceConditioningVideo'), value: 'video' },
      { label: t('widgets.video.referenceConditioningAudio'), value: 'audio' },
    ],
    [t]
  );
  const conditioningCollection = useMemo(() => createListCollection({ items: conditioningItems }), [conditioningItems]);
  const anchorConditioningCollection = useMemo(
    () => createListCollection({ items: conditioningItems.filter((item) => item.value !== 'audio') }),
    [conditioningItems]
  );
  const detailCollection = useMemo(
    () =>
      createListCollection({
        items: [
          { label: t('widgets.video.referenceDetailMax'), value: 'max' },
          { label: t('widgets.video.referenceDetailMatch'), value: 'match' },
        ],
      }),
    [t]
  );
  const collections = useMemo(
    () => ({
      anchorConditioning: anchorConditioningCollection,
      conditioning: conditioningCollection,
      detail: detailCollection,
    }),
    [anchorConditioningCollection, conditioningCollection, detailCollection]
  );
  const handlePickImage = useCallback(() => imageInputRef.current?.click(), []);
  const handlePickVideo = useCallback(() => videoInputRef.current?.click(), []);

  const addImageReference = useCallback(
    (image: { height: number; name: string; width: number }) => {
      setErrorMessage(null);

      // Recheck live capacity after async resolution; normalization would otherwise discard an existing reference
      // on overflow.
      let declined = false;

      onChange((current) => {
        if (current.filter((entry) => entry.kind === 'image').length >= maxImages) {
          declined = true;

          return current;
        }

        return [
          ...current,
          {
            // Derive defaults from the live list because another image may have arrived during the await.
            detail: getDefaultReferenceImageDetail(current),
            image: { height: image.height, image_name: image.name, width: image.width },
            kind: 'image',
          },
        ];
      });
      if (declined) {
        setErrorMessage(t('widgets.video.referenceImageCapRace', { max: maxImages }));
      }
    },
    [maxImages, onChange, t]
  );

  const adoptImageByName = useCallback(
    async (imageName: string) => {
      setIsLoading(true);

      try {
        const item = await galleryItems.resolve({ kind: 'image', name: imageName });

        if (item?.kind === 'image') {
          addImageReference(item);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setErrorMessage(message);
        reportError(message);
      } finally {
        setIsLoading(false);
      }
    },
    [addImageReference, reportError]
  );

  const addVideoItem = useCallback(
    (item: GalleryVideoItem) => {
      // Construct once so the caller retains the same entry identity through reorders.
      const entry = createVideoReferenceEntry(item);
      let declined = false;

      setErrorMessage(null);
      onChange((current) => {
        if (current.filter((existing) => existing.kind === 'video').length >= maxVideos) {
          declined = true;

          return current;
        }

        return [...current, entry];
      });
      if (declined) {
        setErrorMessage(t('widgets.video.referenceVideoCapRace', { max: maxVideos }));

        return null;
      }

      return entry;
    },
    [maxVideos, onChange, t]
  );

  const addVideoReference = useCallback(
    async (videoName: string) => {
      setErrorMessage(null);
      setIsLoading(true);

      try {
        const item = await galleryItems.resolve({ kind: 'video', name: videoName });

        if (item?.kind === 'video') {
          addVideoItem(item);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setErrorMessage(message);
        reportError(message);
      } finally {
        setIsLoading(false);
      }
    },
    [addVideoItem, reportError]
  );

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const item = getSingleGalleryDragItem(event.active.data.current);

      if (isInert || event.over?.id !== DROP_ID || !item) {
        return;
      }
      if (item.kind === 'video' && canAddVideo) {
        void addVideoReference(item.name);
      } else if (item.kind === 'image' && canAddImage) {
        void adoptImageByName(item.name);
      }
    },
    [adoptImageByName, addVideoReference, canAddImage, canAddVideo, isInert]
  );

  useDndMonitor({ onDragEnd: handleDragEnd });

  const uploadFile = useCallback(
    async (file: File, kind: 'image' | 'video') => {
      setErrorMessage(null);
      const owner = captureAccountScope();
      setIsLoading(true);

      try {
        if (kind === 'video') {
          const uploaded = await galleryTransfers.uploadVideo(file, getUploadBoardId(), { signal: owner.signal });

          assertAccountScopeCurrent(owner);
          await addVideoReference(uploaded.name);
        } else {
          const uploaded = await galleryTransfers.upload(file, getUploadBoardId(), { signal: owner.signal });

          assertAccountScopeCurrent(owner);
          addImageReference({ height: uploaded.height, name: uploaded.imageName, width: uploaded.width });
        }
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
    [addImageReference, addVideoReference, getUploadBoardId, reportError, touchGalleryImages]
  );

  const handleImageFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];

      if (file) {
        void uploadFile(file, 'image');
      }
      event.currentTarget.value = '';
    },
    [uploadFile]
  );
  const handleVideoFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];

      if (file) {
        void uploadFile(file, 'video');
      }
      event.currentTarget.value = '';
    },
    [uploadFile]
  );

  const pickerSelection = useMemo<GalleryPickerSelection>(
    () => ({
      addedKeys: new Set(
        references.map((reference) =>
          reference.kind === 'video'
            ? toGalleryItemKey({ kind: 'video', name: reference.clip.video_name })
            : toGalleryItemKey({ kind: 'image', name: reference.image.image_name })
        )
      ),
      mode: 'multiple',
      remaining: { image: Math.max(0, maxImages - imageCount), video: Math.max(0, maxVideos - videoCount) },
    }),
    [imageCount, maxImages, maxVideos, references, videoCount]
  );
  const addPickedVideo = useCallback(
    (item: GalleryVideoItem) => {
      // Apply picks synchronously so subsequent clicks see current capacity and cannot duplicate or reorder
      // pending selections.
      addVideoItem(item);
    },
    [addVideoItem]
  );
  const handlePick = useCallback(
    (item: GalleryItem) => {
      if (item.kind === 'video') {
        addPickedVideo(item);
      } else {
        addImageReference(item);
      }
    },
    [addImageReference, addPickedVideo]
  );

  const updateReference = useCallback(
    (index: number, reference: VideoReferenceItem) => {
      onChange((current) => current.map((entry, entryIndex) => (entryIndex === index ? reference : entry)));
    },
    [onChange]
  );
  const removeReference = useCallback(
    (index: number) => {
      onChange((current) => current.filter((_, entryIndex) => entryIndex !== index));
    },
    [onChange]
  );
  // Key by kind/name occurrence, not index, to preserve focus and video elements across moves. Identical twins
  // reuse instances with swapped props; independent twin identity would require persisted per-entry IDs.
  const referenceKeys = useMemo(() => {
    const seen = new Map<string, number>();

    return references.map((reference) => {
      const name = reference.kind === 'video' ? reference.clip.video_name : reference.image.image_name;
      // Include kind because image and video names may collide.
      const identity = `${reference.kind}:${name}`;
      const occurrence = seen.get(identity) ?? 0;

      seen.set(identity, occurrence + 1);

      return `${identity}-${occurrence}`;
    });
  }, [references]);
  const promptLabels = useMemo(() => referencePromptLabels(references), [references]);

  const moveReference = useCallback(
    (index: number, direction: -1 | 1) => {
      onChange((current) => {
        const target = index + direction;

        if (target < 0 || target >= current.length) {
          return current;
        }
        const next = [...current];
        const [entry] = next.splice(index, 1);

        if (!entry) {
          return current;
        }
        next.splice(target, 0, entry);

        return next;
      });
    },
    [onChange]
  );

  return (
    <Stack gap="2">
      {references.map((reference, index) => (
        <ReferenceCard
          key={referenceKeys[index]}
          audioLabel={promptLabels[index]?.audio ?? null}
          collections={collections}
          disabled={isInert}
          index={index}
          canMoveDown={index < references.length - 1 && index + 1 !== anchorIndex}
          canMoveUp={index > 0 && index !== anchorIndex}
          pictureLabel={promptLabels[index]?.picture ?? null}
          reference={reference}
          targetArea={targetArea}
          videoLabel={promptLabels[index]?.video ?? null}
          onMove={moveReference}
          onRemove={removeReference}
          onUpdate={updateReference}
        />
      ))}

      <Field helpText={t('widgets.video.referencesHelp')} label={t('widgets.video.addReference')}>
        <DropZone
          ref={setNodeRef}
          {...(isInert ? DROP_ZONE_DISABLED_PROPS : {})}
          {...(isLoading ? DROP_ZONE_BUSY_PROPS : {})}
          isDisabled={isInert}
          isOver={isOver && acceptsActiveDrag}
          _focusVisible={DROP_ZONE_FOCUS_PROPS}
          position="relative"
        >
          <Stack gap="1.5" p="2">
            <GalleryPickerPopover
              accept={ACCEPT_MEDIA}
              label={t('widgets.video.chooseReference')}
              selection={pickerSelection}
              onPick={handlePick}
            >
              <Button disabled={isInert || (!canAddImage && !canAddVideo)} size="xs" variant="outline" w="full">
                {isLoading ? <Spinner size="xs" /> : <Icon as={ImagePlusIcon} boxSize="3.5" />}
                {t('widgets.video.chooseReference')}
                <Icon as={ChevronDownIcon} boxSize="3" color="fg.subtle" />
              </Button>
            </GalleryPickerPopover>
            <HStack gap="1" justify="center">
              <Button disabled={isInert || !canAddImage} size="xs" variant="ghost" onClick={handlePickImage}>
                <UploadIcon size={12} />
                {t('widgets.video.uploadImageReference')}
              </Button>
              <Button disabled={isInert || !canAddVideo} size="xs" variant="ghost" onClick={handlePickVideo}>
                <UploadIcon size={12} />
                {t('widgets.video.uploadVideoReference')}
              </Button>
            </HStack>
          </Stack>
          <DropTargetOverlay isActive={acceptsActiveDrag} isOver={isOver} label={t('widgets.video.dropReference')} />
        </DropZone>
      </Field>
      {errorMessage ? (
        <Text color="fg.error" fontSize="xs">
          {errorMessage}
        </Text>
      ) : null}
      <Input accept={IMAGE_UPLOAD_ACCEPT} hidden ref={imageInputRef} type="file" onChange={handleImageFileChange} />
      {/* The server wraps audio uploads as waveform videos. */}
      <Input accept={MEDIA_UPLOAD_ACCEPT} hidden ref={videoInputRef} type="file" onChange={handleVideoFileChange} />
    </Stack>
  );
});

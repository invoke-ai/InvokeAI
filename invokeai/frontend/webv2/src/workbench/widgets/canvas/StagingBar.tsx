import type { CanvasStagingAreaContractV2, CanvasStagingCandidateContract } from '@workbench/canvas-engine/api';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { CanvasStagingSlot } from '@workbench/canvasStagingView';

import {
  Flex,
  HStack,
  Menu,
  Portal,
  ProgressCircle,
  ScrollArea,
  Skeleton,
  Spinner,
  Stack,
  Text,
} from '@chakra-ui/react';
import { galleryDurability } from '@features/gallery';
import { galleryImageUrls } from '@features/gallery/utility';
import { useQueueItemProgress, useQueueItemProgressImage } from '@features/queue/react';
import { Button, IconButton, MenuContent, toaster, Tooltip } from '@platform/ui';
import { StreamingImageFrame } from '@platform/ui/streaming-image/StreamingImageFrame';
import { progressImageToStreamingSource } from '@platform/ui/streaming-image/streamingImageSource';
import { getCancelableCanvasStagingQueueItemId } from '@workbench/canvasStagingView';
import { CanvasOptionsBar } from '@workbench/widgets/canvas/tool-options/CanvasOptionsBar';
import {
  CheckIcon,
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronUpIcon,
  EyeIcon,
  EyeOffIcon,
  SaveIcon,
  SparklesIcon,
  Trash2Icon,
  XIcon,
} from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { CanvasFloatingBarDivider } from './CanvasFloatingBar';

type AutoSwitchMode = CanvasStagingAreaContractV2['autoSwitchMode'];

const THUMBNAIL_STRIP_HEIGHT = '5rem';
const AUTO_SWITCH_MODES: AutoSwitchMode[] = ['off', 'progress', 'latest'];
const MENU_POSITIONING = { placement: 'top-end' } as const;

interface StagingBarProps {
  antialiasProgressImages: boolean;
  areThumbnailsVisible: boolean;
  autoSwitchMode: AutoSwitchMode;
  canAccept: boolean;
  hasMultipleSlots: boolean;
  isGenerating: boolean;
  isVisible: boolean;
  selectedCandidate: CanvasStagingCandidateContract | undefined;
  selectedImageIndex: number;
  selectedSlot: CanvasStagingSlot | undefined;
  slots: CanvasStagingSlot[];
  onAccept: () => void;
  onCancelQueueItem: (queueItemId: string) => void;
  onCycle: (direction: -1 | 1) => void;
  onDiscardAll: () => void;
  onDiscardSelected: () => void;
  onSelectImage: (imageIndex: number) => void;
  onSetAutoSwitch: (mode: AutoSwitchMode) => void;
  onToggleThumbnails: () => void;
  onToggleVisibility: () => void;
}

/**
 * The floating staging bar over the canvas: appears while a canvas generation
 * is in flight or staged candidates await a decision. It drives the reducer's
 * staging actions (cycle / accept / discard / auto-switch); the candidate and
 * live progress pixels themselves are drawn on the engine canvas via
 * `engine.previews.setStagedPreview` (wired in {@link CanvasWidgetView}). Rendered inside
 * the canvas's bottom-center floating group, stacked directly above the tool
 * options bar; positioning is the parent's job.
 */
export const StagingBar = ({
  antialiasProgressImages,
  areThumbnailsVisible,
  autoSwitchMode,
  canAccept,
  hasMultipleSlots,
  isGenerating,
  isVisible,
  selectedCandidate,
  selectedImageIndex,
  selectedSlot,
  slots,
  onAccept,
  onCancelQueueItem,
  onCycle,
  onDiscardAll,
  onDiscardSelected,
  onSelectImage,
  onSetAutoSwitch,
  onToggleThumbnails,
  onToggleVisibility,
}: StagingBarProps) => {
  const { t } = useTranslation();
  const [isSaving, setIsSaving] = useState(false);
  const hasSlots = slots.length > 0;
  const cancelableQueueItemId = getCancelableCanvasStagingQueueItemId(selectedSlot);

  const handleSaveToGallery = async () => {
    if (!selectedCandidate || isSaving) {
      return;
    }
    setIsSaving(true);
    try {
      await galleryDurability.save(selectedCandidate.imageName);
      toaster.create({
        description: t('widgets.canvas.staging.savedDescription', { name: selectedCandidate.imageName }),
        title: t('widgets.canvas.staging.saved'),
        type: 'success',
      });
    } catch {
      toaster.create({ title: t('widgets.canvas.staging.saveError'), type: 'error' });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Stack align="center" gap="2" w="full">
      {hasSlots ? (
        <ScrollArea.Root
          h={areThumbnailsVisible ? THUMBNAIL_STRIP_HEIGHT : '0'}
          opacity={areThumbnailsVisible ? 1 : 0}
          pointerEvents={areThumbnailsVisible ? 'auto' : 'none'}
          size="xs"
          transition="height var(--wb-motion-duration-slow) ease, opacity var(--wb-motion-duration-slow) ease"
          variant="hover"
          w="full"
        >
          <ScrollArea.Viewport h="full" w="full">
            <ScrollArea.Content asChild>
              <HStack h="full" justify="center">
                {slots.map((slot, index) => (
                  <StagingThumbnail
                    key={slot.id}
                    antialiasProgressImages={antialiasProgressImages}
                    index={index}
                    isSelected={index === selectedImageIndex}
                    slot={slot}
                    onSelect={() => onSelectImage(index)}
                  />
                ))}
              </HStack>
            </ScrollArea.Content>
          </ScrollArea.Viewport>
          <ScrollArea.Scrollbar orientation="horizontal">
            <ScrollArea.Thumb />
          </ScrollArea.Scrollbar>
        </ScrollArea.Root>
      ) : null}

      <CanvasOptionsBar>
        {isGenerating ? (
          <HStack color="fg.muted" gap="1.5" px="1">
            <Spinner size="xs" />
            <Text fontSize="xs" fontWeight="600">
              {t('widgets.canvas.staging.generating')}
            </Text>
          </HStack>
        ) : null}

        {hasSlots && selectedSlot ? (
          <>
            <IconButton
              aria-label={
                areThumbnailsVisible
                  ? t('widgets.canvas.hideStagingThumbnails')
                  : t('widgets.canvas.showStagingThumbnails')
              }
              size="xs"
              variant="ghost"
              onClick={onToggleThumbnails}
            >
              {areThumbnailsVisible ? <ChevronDownIcon /> : <ChevronUpIcon />}
            </IconButton>

            <HStack gap="0.5">
              <IconButton
                aria-label={t('widgets.canvas.previousStagedCandidate')}
                disabled={!hasMultipleSlots}
                size="xs"
                variant="ghost"
                onClick={() => onCycle(-1)}
              >
                <ChevronLeftIcon />
              </IconButton>
              <Text fontSize="xs" fontVariantNumeric="tabular-nums" minW="3.5rem" px="1" textAlign="center">
                {t('widgets.canvas.candidateCount', {
                  current: selectedImageIndex + 1,
                  total: slots.length,
                })}
              </Text>
              <IconButton
                aria-label={t('widgets.canvas.nextStagedCandidate')}
                disabled={!hasMultipleSlots}
                size="xs"
                variant="ghost"
                onClick={() => onCycle(1)}
              >
                <ChevronRightIcon />
              </IconButton>
            </HStack>

            <CanvasFloatingBarDivider />

            <AutoSwitchMenu mode={autoSwitchMode} onSelect={onSetAutoSwitch} />

            {cancelableQueueItemId ? (
              <Button size="xs" variant="ghost" onClick={() => onCancelQueueItem(cancelableQueueItemId)}>
                <XIcon />
                {t('common.cancel')}
              </Button>
            ) : null}

            {selectedCandidate ? (
              <>
                <Tooltip
                  content={
                    isVisible
                      ? t('widgets.canvas.hideStagedResultPreview')
                      : t('widgets.canvas.showStagedResultPreview')
                  }
                >
                  <IconButton
                    aria-label={
                      isVisible
                        ? t('widgets.canvas.hideStagedResultPreview')
                        : t('widgets.canvas.showStagedResultPreview')
                    }
                    size="xs"
                    variant="ghost"
                    onClick={onToggleVisibility}
                  >
                    {isVisible ? <EyeIcon /> : <EyeOffIcon />}
                  </IconButton>
                </Tooltip>

                <Tooltip content={t('widgets.canvas.staging.saveToGallery')}>
                  <IconButton
                    aria-label={t('widgets.canvas.staging.saveToGallery')}
                    disabled={isSaving}
                    size="xs"
                    variant="ghost"
                    onClick={handleSaveToGallery}
                  >
                    {isSaving ? <Spinner size="xs" /> : <SaveIcon />}
                  </IconButton>
                </Tooltip>

                <Tooltip content={t('common.discard')}>
                  <IconButton aria-label={t('common.discard')} size="xs" variant="ghost" onClick={onDiscardSelected}>
                    <XIcon />
                  </IconButton>
                </Tooltip>

                <CanvasFloatingBarDivider />

                <Button size="xs" variant="ghost" onClick={onDiscardAll}>
                  <Trash2Icon />
                  {t('common.discardAll')}
                </Button>

                <Button disabled={!canAccept} size="xs" onClick={onAccept}>
                  <CheckIcon />
                  {t('widgets.canvas.acceptToLayer')}
                </Button>
              </>
            ) : null}
          </>
        ) : null}
      </CanvasOptionsBar>
    </Stack>
  );
};

const AutoSwitchMenu = ({ mode, onSelect }: { mode: AutoSwitchMode; onSelect: (mode: AutoSwitchMode) => void }) => {
  const { t } = useTranslation();
  const label = (value: AutoSwitchMode): string =>
    t(
      value === 'off'
        ? 'widgets.canvas.staging.autoSwitchOff'
        : value === 'progress'
          ? 'widgets.canvas.staging.autoSwitchProgress'
          : 'widgets.canvas.staging.autoSwitchLatest'
    );

  return (
    <Menu.Root positioning={MENU_POSITIONING}>
      <Tooltip content={t('widgets.canvas.staging.autoSwitch')}>
        <span style={{ display: 'inline-flex' }}>
          <Menu.Trigger asChild>
            <Button minW="unset" px="2" size="xs" variant="ghost">
              <SparklesIcon size={13} />
              <Text fontSize="xs">{label(mode)}</Text>
            </Button>
          </Menu.Trigger>
        </span>
      </Tooltip>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="8rem" py="1">
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel>{t('widgets.canvas.staging.autoSwitch')}</Menu.ItemGroupLabel>
              {AUTO_SWITCH_MODES.map((value) => (
                <Menu.Item key={value} value={value} onClick={() => onSelect(value)}>
                  <CheckIcon size={12} opacity={mode === value ? 1 : 0} />
                  <Menu.ItemText fontSize="xs">{label(value)}</Menu.ItemText>
                </Menu.Item>
              ))}
            </Menu.ItemGroup>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const StagingThumbnail = ({
  antialiasProgressImages,
  index,
  isSelected,
  slot,
  onSelect,
}: {
  antialiasProgressImages: boolean;
  index: number;
  isSelected: boolean;
  slot: CanvasStagingSlot;
  onSelect: () => void;
}) => {
  const { t } = useTranslation();

  return (
    <Stack
      aria-label={t('widgets.canvas.selectStagedCandidate', { number: index + 1 })}
      as="button"
      bg="bg.emphasized"
      borderWidth="2px"
      borderColor={isSelected ? 'accent.solid' : 'border.subtle'}
      flex="0 0 auto"
      h="full"
      overflow="hidden"
      position="relative"
      rounded="md"
      shadow={isSelected ? '0 0 0 1px {colors.accent.solid}' : 'md'}
      style={{ aspectRatio: '1 / 1' }}
      onClick={onSelect}
    >
      {slot.kind === 'candidate' ? (
        <img
          alt={slot.candidate.imageName}
          src={slot.candidate.thumbnailUrl || galleryImageUrls.thumbnail(slot.candidate.imageName)}
          style={{ display: 'block', height: '100%', objectFit: 'cover', width: '100%' }}
        />
      ) : (
        <StagingPlaceholderThumbnail antialiasProgressImages={antialiasProgressImages} slot={slot} />
      )}
      <Text
        bg="blackAlpha.700"
        bottom="1"
        color="white"
        fontSize="2xs"
        fontWeight="700"
        left="1"
        px="1.5"
        position="absolute"
        rounded="sm"
      >
        {index + 1}
      </Text>
    </Stack>
  );
};

const StagingPlaceholderThumbnail = ({
  antialiasProgressImages,
  slot,
}: {
  antialiasProgressImages: boolean;
  slot: Extract<CanvasStagingSlot, { kind: 'placeholder' }>;
}) => {
  const progressImage = useQueueItemProgressImage(slot.queueItemId, slot.itemIndex);
  const progress = useQueueItemProgress(slot.queueItemId);
  const isActive = progress?.activeItemIndex === slot.itemIndex;
  const percentage = typeof progress?.percentage === 'number' ? Math.round(progress.percentage * 100) : null;

  return (
    <>
      <StreamingImageFrame
        fit="cover"
        h="full"
        liveImage={progressImageToStreamingSource(progressImage)}
        shouldAntialiasLiveImage={antialiasProgressImages}
        w="full"
      >
        <Skeleton h="full" w="full" />
      </StreamingImageFrame>
      {isActive ? <StagingPlaceholderProgress percentage={percentage} /> : null}
    </>
  );
};

const StagingPlaceholderProgress = ({ percentage }: { percentage: number | null }) => {
  const { t } = useTranslation();

  return (
    <Flex
      align="center"
      aria-label={
        percentage === null
          ? t('widgets.gallery.generationProgress')
          : t('widgets.gallery.generationProgressPercent', { percentage })
      }
      inset="0"
      justify="center"
      pointerEvents="none"
      position="absolute"
      zIndex="1"
    >
      <ProgressCircle.Root bg="bg/85" borderWidth={1} p={0.5} rounded="full" size="xs" value={percentage}>
        <ProgressCircle.Circle>
          <ProgressCircle.Track />
          <ProgressCircle.Range />
        </ProgressCircle.Circle>
      </ProgressCircle.Root>
    </Flex>
  );
};

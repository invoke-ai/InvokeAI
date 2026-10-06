import type { CanvasStagingAreaContractV2, CanvasStagingCandidateContract } from '@workbench/canvas-engine/api';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { CanvasStagingSlot } from '@workbench/canvasStagingView';

import {
  Box,
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
import { galleryImageUrls } from '@features/gallery/utility';
import { useQueueItemProgress, useQueueItemProgressImage } from '@features/queue/react';
import { Button, Group, IconButton, MenuContent, Tooltip, useTooltipTriggerIds } from '@platform/ui';
import { StreamingImageFrame } from '@platform/ui/streaming-image/StreamingImageFrame';
import { progressImageToStreamingSource } from '@platform/ui/streaming-image/streamingImageSource';
import { wheelScrollsHorizontally } from '@platform/ui/wheelScrollsHorizontally';
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
import { useCallback, useState, type MouseEvent as ReactMouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { CanvasFloatingBarDivider } from './CanvasFloatingBar';
import { StagingItemContextMenu, type StagingItemContextMenuTarget } from './StagingItemContextMenu';

type AutoSwitchMode = CanvasStagingAreaContractV2['autoSwitchMode'];

const THUMBNAIL_STRIP_HEIGHT = '5rem';
const AUTO_SWITCH_MODES: AutoSwitchMode[] = ['off', 'progress', 'latest'];
const MENU_POSITIONING = { placement: 'top-end' } as const;
/**
 * Below this bar width the "Generating" text and "Discard All" collapse to icons, so the accept split button (the
 * only pointer path to "Keep as Hidden Layer") stays inside the canvas. Measured on the staging slot itself.
 */
const COMPACT = '@container staging-bar (width < 48rem)';
/**
 * Below this the remaining text goes too (the candidate counter, the auto-switch mode, Cancel and the accept label),
 * leaving icons whose names and tooltips carry the words, so the bar fits a canvas down to about 360px.
 */
const NARROW = '@container staging-bar (width < 36rem)';
const STAGING_ROOT_CSS = { containerName: 'staging-bar', containerType: 'inline-size' } as const;
const WIDE_ONLY_CSS = { [COMPACT]: { display: 'none' } } as const;
const COMPACT_ONLY_CSS = { display: 'none', [COMPACT]: { display: 'inline-flex' } } as const;
const SR_ONLY = {
  clipPath: 'inset(50%)',
  height: '1px',
  overflow: 'hidden',
  position: 'absolute',
  whiteSpace: 'nowrap',
  width: '1px',
} as const;
/** Visually hidden while compact, still read as the status text. */
const COMPACT_SR_ONLY_CSS = { [COMPACT]: SR_ONLY } as const;
/** Visually hidden while narrow; still the element's text for assistive technology. */
const NARROW_SR_ONLY_CSS = { [NARROW]: SR_ONLY } as const;
/** Gone while narrow: group rules (the icons' spacing still separates the groups) and labelled buttons. */
const NARROW_HIDDEN_CSS = { [NARROW]: { display: 'none' } } as const;
const NARROW_ONLY_CSS = { display: 'none', [NARROW]: { display: 'inline-flex' } } as const;
/** Buttons whose text goes while narrow keep a square icon footprint. */
const NARROW_ICON_BUTTON_CSS = { [NARROW]: { minW: '8', px: '0' } } as const;

interface StagingBarProps {
  antialiasProgressImages: boolean;
  areThumbnailsVisible: boolean;
  autoSwitchMode: AutoSwitchMode;
  canAccept: boolean;
  hasMultipleSlots: boolean;
  isGenerating: boolean;
  /** A gallery save of the selected candidate is in flight. */
  isSavingToGallery: boolean;
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
  onPreloadCandidate: (imageName: string) => void;
  onSelectImage: (imageIndex: number) => void;
  /** Saves the candidate to the Gallery and reports where it went. */
  onSaveToGallery: (imageName: string) => Promise<void>;
  onSaveToLayerAndContinue: () => void;
  onSetAutoSwitch: (mode: AutoSwitchMode) => void;
  onToggleThumbnails: () => void;
  onToggleVisibility: () => void;
}

/**
 * Show staging controls for running or pending canvas results. {@link CanvasWidgetView} feeds pixels to
 * engine.previews; the parent positions this bar above tool options.
 */
export const StagingBar = ({
  antialiasProgressImages,
  areThumbnailsVisible,
  autoSwitchMode,
  canAccept,
  hasMultipleSlots,
  isGenerating,
  isSavingToGallery,
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
  onPreloadCandidate,
  onSelectImage,
  onSaveToGallery,
  onSaveToLayerAndContinue,
  onSetAutoSwitch,
  onToggleThumbnails,
  onToggleVisibility,
}: StagingBarProps) => {
  const { t } = useTranslation();
  const [contextMenuTarget, setContextMenuTarget] = useState<StagingItemContextMenuTarget | null>(null);
  const closeContextMenu = useCallback(() => setContextMenuTarget(null), []);
  // The menu's candidate actions act on the selection, so it only stays open
  // while its slot is the selected one: an auto-switch to a newer result
  // closes it rather than letting Discard hit the wrong candidate.
  if (contextMenuTarget && selectedSlot?.id !== contextMenuTarget.slot.id) {
    setContextMenuTarget(null);
  }
  const hasSlots = slots.length > 0;
  const cancelableQueueItemId = getCancelableCanvasStagingQueueItemId(selectedSlot);
  const acceptLabel = t('widgets.canvas.acceptToLayer');
  const acceptMenuIds = useTooltipTriggerIds();
  const thumbnailsLabel = areThumbnailsVisible
    ? t('widgets.canvas.hideStagingThumbnails')
    : t('widgets.canvas.showStagingThumbnails');

  const handleSaveToGallery = () => {
    if (selectedCandidate && !isSavingToGallery) {
      void onSaveToGallery(selectedCandidate.imageName);
    }
  };

  return (
    <Stack
      align="center"
      animationDuration="moderate"
      animationName="slide-from-bottom, fade-in"
      css={STAGING_ROOT_CSS}
      gap="2"
      w="full"
    >
      {contextMenuTarget ? (
        <StagingItemContextMenu
          canAccept={canAccept}
          target={contextMenuTarget}
          onAccept={onAccept}
          onClose={closeContextMenu}
          onDiscard={onDiscardSelected}
          onSaveToGallery={handleSaveToGallery}
        />
      ) : null}
      {hasSlots ? (
        // Constrain thumbnail scrolling to canvas width so centered overflow cannot be clipped outside the
        // surface.
        <ScrollArea.Root
          h={areThumbnailsVisible ? THUMBNAIL_STRIP_HEIGHT : '0'}
          opacity={areThumbnailsVisible ? 1 : 0}
          pointerEvents={areThumbnailsVisible ? 'auto' : 'none'}
          transition="height var(--wb-motion-duration-slow) ease, opacity var(--wb-motion-duration-slow) ease"
          variant="hover"
          w="full"
        >
          <ScrollArea.Viewport ref={wheelScrollsHorizontally} h="full" scrollPaddingInline="2" w="full">
            {/*
             * Leave the content slot's width alone: its inline `min-width:
             * fit-content` grows it to hold every thumbnail, so `justify` only
             * centers when the strip fits and no thumbnail ever lands at a
             * negative offset that scrolling cannot reach.
             */}
            <ScrollArea.Content asChild>
              <HStack h="full" justify="center">
                {slots.map((slot, index) => (
                  <StagingThumbnail
                    key={slot.id}
                    antialiasProgressImages={antialiasProgressImages}
                    index={index}
                    isSelected={index === selectedImageIndex}
                    slot={slot}
                    onContextMenu={(event) => {
                      if (slot.kind !== 'candidate') {
                        return;
                      }
                      // The menu acts on the selected candidate, so a right-click selects first.
                      // The keyboard's context-menu key reports no pointer; anchor to the thumbnail.
                      event.preventDefault();
                      onSelectImage(index);
                      const rect = event.currentTarget.getBoundingClientRect();
                      const fromKeyboard = event.clientX === 0 && event.clientY === 0;
                      setContextMenuTarget({
                        slot,
                        x: fromKeyboard ? rect.left + rect.width / 2 : event.clientX,
                        y: fromKeyboard ? rect.top : event.clientY,
                      });
                    }}
                    onPreloadCandidate={onPreloadCandidate}
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
          <Tooltip content={t('widgets.canvas.staging.generating')}>
            <HStack color="fg.muted" flexShrink="0" gap="1.5" position="relative" px="1" role="status">
              <Spinner />
              <Text css={COMPACT_SR_ONLY_CSS} fontSize="md" fontWeight="600" whiteSpace="nowrap">
                {t('widgets.canvas.staging.generating')}
              </Text>
            </HStack>
          </Tooltip>
        ) : null}

        {hasSlots && selectedSlot ? (
          <>
            <Tooltip content={thumbnailsLabel}>
              <IconButton aria-label={thumbnailsLabel} variant="ghost" onClick={onToggleThumbnails}>
                {areThumbnailsVisible ? <ChevronDownIcon /> : <ChevronUpIcon />}
              </IconButton>
            </Tooltip>

            <HStack gap="0.5">
              <Tooltip content={t('widgets.canvas.previousStagedCandidate')}>
                <IconButton
                  aria-label={t('widgets.canvas.previousStagedCandidate')}
                  disabled={!hasMultipleSlots}
                  variant="ghost"
                  onClick={() => onCycle(-1)}
                >
                  <ChevronLeftIcon />
                </IconButton>
              </Tooltip>
              <Text
                css={NARROW_SR_ONLY_CSS}
                fontSize="md"
                fontVariantNumeric="tabular-nums"
                minW="3.5rem"
                px="1"
                textAlign="center"
              >
                {t('widgets.canvas.candidateCount', {
                  current: selectedImageIndex + 1,
                  total: slots.length,
                })}
              </Text>
              <Tooltip content={t('widgets.canvas.nextStagedCandidate')}>
                <IconButton
                  aria-label={t('widgets.canvas.nextStagedCandidate')}
                  disabled={!hasMultipleSlots}
                  variant="ghost"
                  onClick={() => onCycle(1)}
                >
                  <ChevronRightIcon />
                </IconButton>
              </Tooltip>
            </HStack>

            <CanvasFloatingBarDivider css={NARROW_HIDDEN_CSS} />

            <AutoSwitchMenu mode={autoSwitchMode} onSelect={onSetAutoSwitch} />

            {cancelableQueueItemId ? (
              <>
                {/* Labelled while there is room; an icon with the tooltip once narrow, as Discard All does. */}
                <Button
                  css={NARROW_HIDDEN_CSS}
                  flexShrink="0"
                  variant="ghost"
                  onClick={() => onCancelQueueItem(cancelableQueueItemId)}
                >
                  <XIcon />
                  {t('common.cancel')}
                </Button>
                <Tooltip content={t('common.cancel')}>
                  <IconButton
                    aria-label={t('common.cancel')}
                    css={NARROW_ONLY_CSS}
                    variant="ghost"
                    onClick={() => onCancelQueueItem(cancelableQueueItemId)}
                  >
                    <XIcon />
                  </IconButton>
                </Tooltip>
              </>
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
                    variant="ghost"
                    onClick={onToggleVisibility}
                  >
                    {isVisible ? <EyeIcon /> : <EyeOffIcon />}
                  </IconButton>
                </Tooltip>

                <Tooltip content={t('widgets.canvas.staging.saveToGallery')}>
                  <IconButton
                    aria-label={t('widgets.canvas.staging.saveToGallery')}
                    disabled={isSavingToGallery}
                    variant="ghost"
                    onClick={handleSaveToGallery}
                  >
                    {isSavingToGallery ? <Spinner /> : <SaveIcon />}
                  </IconButton>
                </Tooltip>

                <Tooltip content={t('common.discard')}>
                  <IconButton aria-label={t('common.discard')} variant="ghost" onClick={onDiscardSelected}>
                    <XIcon />
                  </IconButton>
                </Tooltip>

                <CanvasFloatingBarDivider css={NARROW_HIDDEN_CSS} />

                <Button css={WIDE_ONLY_CSS} flexShrink="0" variant="ghost" onClick={onDiscardAll}>
                  <Trash2Icon />
                  {t('common.discardAll')}
                </Button>
                <Tooltip content={t('common.discardAll')}>
                  <IconButton
                    aria-label={t('common.discardAll')}
                    css={COMPACT_ONLY_CSS}
                    variant="ghost"
                    onClick={onDiscardAll}
                  >
                    <Trash2Icon />
                  </IconButton>
                </Tooltip>

                <Menu.Root ids={acceptMenuIds} positioning={MENU_POSITIONING}>
                  <Group attached flexShrink="0">
                    <Tooltip content={acceptLabel}>
                      <Button
                        aria-label={acceptLabel}
                        css={NARROW_ICON_BUTTON_CSS}
                        disabled={!canAccept}
                        onClick={onAccept}
                      >
                        <CheckIcon />
                        <Box as="span" css={NARROW_SR_ONLY_CSS}>
                          {acceptLabel}
                        </Box>
                      </Button>
                    </Tooltip>
                    <Tooltip content={t('widgets.canvas.staging.moreAcceptOptions')} ids={acceptMenuIds}>
                      <Menu.Trigger asChild>
                        <IconButton
                          aria-label={t('widgets.canvas.staging.moreAcceptOptions')}
                          disabled={!canAccept}
                          minW="0"
                          w="6"
                        >
                          <ChevronDownIcon />
                        </IconButton>
                      </Menu.Trigger>
                    </Tooltip>
                  </Group>
                  <Portal>
                    <Menu.Positioner>
                      <MenuContent minW="13rem" py="1">
                        <Menu.Item value="save-disabled-layer" onClick={onSaveToLayerAndContinue}>
                          <EyeOffIcon size={14} />
                          <Menu.ItemText fontSize="md">{t('widgets.canvas.staging.saveAsDisabledLayer')}</Menu.ItemText>
                        </Menu.Item>
                      </MenuContent>
                    </Menu.Positioner>
                  </Portal>
                </Menu.Root>
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
            <Button
              aria-label={`${t('widgets.canvas.staging.autoSwitch')}: ${label(mode)}`}
              css={NARROW_ICON_BUTTON_CSS}
              minW="unset"
              px="2"
              variant="ghost"
            >
              <SparklesIcon size={13} />
              <Text css={NARROW_SR_ONLY_CSS} fontSize="md">
                {label(mode)}
              </Text>
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
                  <Menu.ItemText fontSize="md">{label(value)}</Menu.ItemText>
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
  onContextMenu,
  onPreloadCandidate,
  onSelect,
}: {
  antialiasProgressImages: boolean;
  index: number;
  isSelected: boolean;
  slot: CanvasStagingSlot;
  onContextMenu: (event: ReactMouseEvent<HTMLElement>) => void;
  onPreloadCandidate: (imageName: string) => void;
  onSelect: () => void;
}) => {
  const { t } = useTranslation();
  // Selection-sensitive ref callbacks scroll the active thumbnail into view without an effect.
  const scrollIntoView = useCallback(
    (node: HTMLElement | null) => {
      if (node && isSelected) {
        node.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      }
    },
    [isSelected]
  );
  const preloadCandidate = useCallback(() => {
    // Once a candidate's small thumbnail settles, warm its full bytes in the
    // engine so cycling to it only pays the bitmap decode. The cache promotes a
    // selected candidate out of the background queue when necessary.
    if (slot.kind === 'candidate') {
      onPreloadCandidate(slot.candidate.imageName);
    }
  }, [onPreloadCandidate, slot]);

  return (
    <Stack
      ref={scrollIntoView}
      aria-current={isSelected || undefined}
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
      onContextMenu={onContextMenu}
    >
      {slot.kind === 'candidate' ? (
        <img
          alt={slot.candidate.imageName}
          src={slot.candidate.thumbnailUrl || galleryImageUrls.thumbnail(slot.candidate.imageName)}
          style={{ display: 'block', height: '100%', objectFit: 'cover', width: '100%' }}
          onError={preloadCandidate}
          onLoad={preloadCandidate}
        />
      ) : (
        <StagingPlaceholderThumbnail antialiasProgressImages={antialiasProgressImages} slot={slot} />
      )}
      <Text
        bg="blackAlpha.700"
        bottom="1"
        color="white"
        fontSize="xs"
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
    <Flex align="center" inset="0" justify="center" pointerEvents="none" position="absolute" zIndex="1">
      <ProgressCircle.Root
        aria-label={
          percentage === null
            ? t('widgets.gallery.generationProgress')
            : t('widgets.gallery.generationProgressPercent', { percentage })
        }
        bg="bg/85"
        borderWidth={1}
        p={0.5}
        rounded="full"
        value={percentage}
      >
        <ProgressCircle.Circle>
          <ProgressCircle.Track />
          <ProgressCircle.Range />
        </ProgressCircle.Circle>
      </ProgressCircle.Root>
    </Flex>
  );
};

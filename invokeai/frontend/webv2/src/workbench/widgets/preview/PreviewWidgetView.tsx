import type { QueueItem } from '@features/queue/contracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Box, Flex, SimpleGrid, Stack, Text } from '@chakra-ui/react';
import { useDndMonitor, type DragEndEvent } from '@dnd-kit/core';
import {
  galleryImages,
  type GalleryBoard,
  type GalleryImage,
  type GalleryImageItem,
  type GalleryItem,
  type GalleryItemKey,
  type GalleryItemRef,
} from '@features/gallery';
import {
  getGalleryCompareImage,
  getGalleryGenerationSequence,
  getGalleryLiveSlots,
  getGallerySelectedImageQuery,
  getGallerySettings,
  getSelectedGalleryItemFromValues,
  getBoundedRecentImages,
  galleryImageItemToGalleryImage,
  isGalleryImageItem,
  legacyGeneratedImageToGalleryItem,
  normalizeGalleryImage,
  toGalleryItemKey,
  toGalleryItemRef,
  type GalleryQueuePlaceholder,
} from '@features/gallery/contracts';
import { galleryBoardsOptions } from '@features/gallery/queries';
import { createGenerateFormValuesSelector } from '@features/generation/react';
import { getDeterminateProgressPercent } from '@features/queue/contracts';
import { useDeviceLabel } from '@features/queue/devices';
import {
  useActiveProgressTarget,
  useActiveProgressTargets,
  useItemProgress,
  useProgressImage,
  useQueueItemProgressImage,
  type LatestProgressImageSnapshot,
} from '@features/queue/react';
import {
  imageUrlToStreamingSource,
  progressImageToStreamingSource,
} from '@platform/ui/streaming-image/streamingImageSource';
import { useStreamingImageSource } from '@platform/ui/streaming-image/useStreamingImageSource';
import { useQuery } from '@tanstack/react-query';
import {
  ImageContextMenu,
  useDeletionConfirmation,
  useImageActions,
  type ImageActions,
  type ImageContextMenuTarget,
} from '@workbench/image-actions';
import { getProjectWidgetValues } from '@workbench/widgetState';
import {
  useActiveProjectId,
  useActiveProjectSelector,
  useWidgetValuesSelector,
  useWorkbenchCommands,
  useWorkbenchQueries,
} from '@workbench/WorkbenchContext';
import { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState, type ReactNode, type Ref } from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewLoupeControls } from './usePreviewLoupe';

import { PreviewCompare } from './PreviewCompare';
import { resolvePreviewCompareDrop } from './previewCompareDnd';
import { usePreviewDensity, type PreviewDensity } from './previewDensity';
import { PreviewFilmstrip } from './PreviewFilmstrip';
import { PreviewFooter, type PreviewFooterMedia } from './PreviewFooter';
import {
  PreviewFrame,
  type PreviewMediaSource,
  type PreviewVideoFrameController,
  type PreviewVideoFrameCopyResult,
} from './PreviewFrame';
import { previewHeaderStore } from './previewHeaderStore';
import {
  getPreviewComparisonMode,
  getPreviewFilmstripVisible,
  getPreviewMetadataOpen,
  type PreviewComparisonMode,
} from './previewSettings';
import { usePreviewNavigation } from './usePreviewNavigation';

/** For the live footer's Details slot, which renders disabled and never fires. */
const noop = (): void => {};

const VIDEO_FRAME_COPY_FAILURE_KEYS = {
  'clipboard-failed': 'widgets.preview.copyCurrentFrameWriteFailed',
  'draw-failed': 'widgets.preview.copyCurrentFrameDrawFailed',
  'encode-failed': 'widgets.preview.copyCurrentFrameEncodeFailed',
  'not-ready': 'widgets.preview.copyCurrentFrameNotReady',
  stale: 'widgets.preview.copyCurrentFrameStale',
  unsupported: 'widgets.preview.copyCurrentFrameUnsupported',
} as const;

export const getVideoFrameCopyNotice = (
  result: PreviewVideoFrameCopyResult,
  translate: (key: string) => string
): { kind: 'error' | 'success'; title: string } =>
  result.ok
    ? { kind: 'success', title: translate('widgets.preview.copyCurrentFrameSuccess') }
    : { kind: 'error', title: translate(VIDEO_FRAME_COPY_FAILURE_KEYS[result.reason]) };

const fallbackBoards: GalleryBoard[] = [
  {
    archived: false,
    assetCount: 0,
    id: 'none',
    imageCount: 0,
    kind: 'uncategorized',
    name: '',
    projectId: null,
    videoCount: 0,
  },
];

const getLocalGalleryItems = (values: Record<string, unknown>, queueItems: QueueItem[]): GalleryImageItem[] => {
  const queueBoardIds = new Map(queueItems.map((item) => [item.id, item.snapshot.galleryBoardId ?? 'none'] as const));

  return getBoundedRecentImages(values.recentImages).map((image) =>
    legacyGeneratedImageToGalleryItem(normalizeGalleryImage(image, queueBoardIds.get(image.sourceQueueItemId)))
  );
};

const getSelectedItem = (values: Record<string, unknown>, localItems: GalleryImageItem[]): GalleryItem | null => {
  const selectedItem = getSelectedGalleryItemFromValues(values);

  if (selectedItem) {
    const selectedItemKey = toGalleryItemKey(selectedItem);
    return localItems.find((candidate) => toGalleryItemKey(candidate) === selectedItemKey) ?? selectedItem;
  }

  return localItems[0] ?? null;
};

const getBoardName = (
  boards: GalleryBoard[],
  boardId: string,
  uncategorizedLabel: string,
  unknownBoardLabel: string
): string =>
  boardId === 'none' ? uncategorizedLabel : (boards.find((board) => board.id === boardId)?.name ?? unknownBoardLabel);

export const getMatchingProgressImage = (
  progressImage: LatestProgressImageSnapshot | null,
  placeholder: GalleryQueuePlaceholder | null
): LatestProgressImageSnapshot | null => {
  if (
    !progressImage?.target ||
    !placeholder ||
    progressImage.target.queueItemId !== placeholder.queueItemId ||
    progressImage.target.itemIndex !== placeholder.itemIndex
  ) {
    return null;
  }

  return progressImage;
};

const selectGenerateRecallValues = createGenerateFormValuesSelector();

/**
 * Bottom padding the grid surface reserves for the overlaid details island —
 * its collapsed height plus the 2-unit inset and gap. The filmstrip
 * deliberately reserves nothing: it is a floating overlay above the media's
 * lower edge, so toggling it never reflows the fitted image. An expanded
 * Details panel grows over the grid on purpose; it is self-capped at `40cqh`
 * and closes back down.
 */
const PREVIEW_OVERLAY_RESERVE = '5.5rem';

export const PreviewWidgetView = ({ region, runtime }: WidgetViewProps) => {
  const galleryValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'gallery'));
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const previewValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'preview'));
  const generateValues = useWidgetValuesSelector('generate', selectGenerateRecallValues);
  const { antialiasProgressImages, showProgressImagesInViewer } = useActiveProjectSelector(
    (project) => project.settings
  );
  const progressImage = useProgressImage();
  const activeProgressTarget = useActiveProgressTarget();
  const activeProgressTargets = useActiveProgressTargets();
  const { account, gallery, notifications, widgets } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const { density, rootRef } = usePreviewDensity(region);
  const recentImages = galleryValues.recentImages;
  const localItems = useMemo(() => getLocalGalleryItems({ recentImages }, queueItems), [queueItems, recentImages]);
  const selectedItem = useMemo(() => getSelectedItem(galleryValues, localItems), [galleryValues, localItems]);
  const compareImage = getGalleryCompareImage(galleryValues);
  const comparisonMode = getPreviewComparisonMode(previewValues);
  const displayBoardId = selectedItem?.boardId ?? 'none';
  const hasSelectedItem = selectedItem !== null;
  const { imageOrderDir, starredFirst } = getGallerySettings(galleryValues);
  const selectedImageQuery = getGallerySelectedImageQuery(galleryValues);
  const selectedItemKey = selectedItem ? toGalleryItemKey(selectedItem) : null;
  const isComparing =
    selectedItem?.kind === 'image' &&
    compareImage !== null &&
    toGalleryItemKey({ kind: 'image', name: compareImage.imageName }) !== selectedItemKey;
  const generationSequence = useMemo(
    () => getGalleryGenerationSequence(queueItems, activeProgressTarget),
    [activeProgressTarget, queueItems]
  );
  const activeGalleryPlaceholder = generationSequence.liveSlot;
  // Multi-GPU runs one session per GPU, so several slots can be live at once. One
  // live slot keeps the existing single-frame preview; two or more are tiled.
  const liveGalleryPlaceholders = useMemo(
    () => getGalleryLiveSlots(generationSequence.chronologicalSlots, activeProgressTargets),
    [activeProgressTargets, generationSequence.chronologicalSlots]
  );
  const matchingProgressImage = getMatchingProgressImage(progressImage, activeGalleryPlaceholder);
  const shouldFollowLive = showProgressImagesInViewer && activeGalleryPlaceholder !== null && !isComparing;
  const { t } = useTranslation();
  const loupeControlsRef = useRef<PreviewLoupeControls | null>(null);
  const videoControllerRef = useRef<PreviewVideoFrameController | null>(null);
  const [copyAvailableItemKey, setCopyAvailableItemKey] = useState<GalleryItemKey | null>(null);

  const boardsQuery = useQuery({
    ...galleryBoardsOptions(),
    enabled: hasSelectedItem,
  });
  const boards = boardsQuery.data ?? fallbackBoards;
  const boardName = getBoardName(
    boards,
    displayBoardId,
    t('widgets.gallery.uncategorized'),
    t('widgets.gallery.unknownBoard')
  );

  const enableLiveFollow = useCallback(
    () => account.updateProjectPreferences({ showProgressImagesInViewer: true }),
    [account]
  );
  const selectGalleryItemAtPage = useCallback(
    (item: GalleryItem, selectionPage: number) => gallery.selectItem(item, undefined, selectionPage, true),
    [gallery]
  );
  const {
    boardItems,
    handleNavigationKeyDown,
    isLoadingBoard,
    navigate,
    navigationCursor,
    navigationQueryKey,
    navigationSequence,
    selectPreviewItem,
  } = usePreviewNavigation({
    activePlaceholder: activeGalleryPlaceholder,
    enableLiveFollow,
    imageOrderDir,
    isComparing,
    localItems,
    queueItems,
    selectGalleryItem: selectGalleryItemAtPage,
    selectedImageQuery,
    selectedItem,
    selectedItemKey,
    shouldFollowLive,
    starredFirst,
  });

  const [contextMenuTarget, setContextMenuTarget] = useState<ImageContextMenuTarget | null>(null);
  const getItemActionContext = useCallback(
    () => ({
      filterIdentity: navigationQueryKey,
      items: boardItems,
      loadOrderedRefs: (signal: AbortSignal) => {
        signal.throwIfAborted();
        return Promise.resolve(boardItems.map(toGalleryItemRef));
      },
      selectedItemKey,
    }),
    [boardItems, navigationQueryKey, selectedItemKey]
  );
  const projectId = useActiveProjectId();
  const { dialog: deletionConfirmationDialog, requestDeletionConfirmation } = useDeletionConfirmation();
  const imageActions = useImageActions({
    boards,
    generateValues,
    getItemActionContext,
    projectId,
    requestDeletionConfirmation,
  });
  const contextMenuItem = useMemo<GalleryItem | null>(() => {
    if (!selectedItem) {
      return null;
    }

    return boardItems.find((item) => toGalleryItemKey(item) === selectedItemKey) ?? selectedItem;
  }, [boardItems, selectedItem, selectedItemKey]);
  const actionImage = useMemo<GalleryImage | null>(
    () =>
      contextMenuItem && isGalleryImageItem(contextMenuItem) ? galleryImageItemToGalleryImage(contextMenuItem) : null,
    [contextMenuItem]
  );
  const exitCompare = useCallback(() => gallery.setCompareItem(null), [gallery]);
  const swapCompareImages = useCallback(() => {
    if (selectedItem?.kind === 'image' && compareImage) {
      selectPreviewItem(legacyGeneratedImageToGalleryItem(compareImage));
      gallery.setCompareItem(selectedItem);
    }
  }, [compareImage, gallery, selectPreviewItem, selectedItem]);
  const isItemCurrent = useCallback(
    (itemKey: GalleryItemKey) => {
      const currentValues = getProjectWidgetValues(queries.getSnapshot().activeProject, 'gallery');
      const currentItem = getSelectedGalleryItemFromValues(currentValues);

      return currentItem !== null && toGalleryItemKey(currentItem) === itemKey;
    },
    [queries]
  );
  const setComparisonMode = useCallback(
    (comparisonMode: PreviewComparisonMode) => widgets.patchValues('preview', { comparisonMode }),
    [widgets]
  );
  const isMetadataOpen = getPreviewMetadataOpen(previewValues);
  const toggleMetadata = useCallback(
    () => widgets.patchValues('preview', { metadataOpen: !isMetadataOpen }),
    [isMetadataOpen, widgets]
  );
  const openVideoDetails = useCallback(() => widgets.patchValues('preview', { metadataOpen: true }), [widgets]);
  const handleVideoCopyAvailabilityChange = useCallback((itemKey: GalleryItemKey, isAvailable: boolean) => {
    setCopyAvailableItemKey((current) => (isAvailable ? itemKey : current === itemKey ? null : current));
  }, []);
  const isVideoFrameCopyAvailable =
    contextMenuItem?.kind === 'video' && copyAvailableItemKey === toGalleryItemKey(contextMenuItem);
  const copyCurrentVideoFrame = useCallback(() => {
    const run = async (): Promise<void> => {
      const controller = videoControllerRef.current;
      let result: PreviewVideoFrameCopyResult;

      if (
        contextMenuItem?.kind !== 'video' ||
        !controller ||
        controller.itemKey !== toGalleryItemKey(contextMenuItem)
      ) {
        result = { ok: false, reason: 'stale' };
      } else {
        try {
          result = await controller.copyCurrentFrame();
        } catch {
          result = { ok: false, reason: 'clipboard-failed' };
        }
      }

      notifications.add(getVideoFrameCopyNotice(result, t));
    };

    void run();
  }, [contextMenuItem, notifications, t]);
  const previewVideoContextActions = useMemo(
    () =>
      contextMenuItem?.kind === 'video'
        ? {
            isCopyCurrentFrameAvailable: isVideoFrameCopyAvailable,
            itemKey: toGalleryItemKey(contextMenuItem),
            onCopyCurrentFrame: copyCurrentVideoFrame,
            onOpenDetails: openVideoDetails,
          }
        : undefined,
    [contextMenuItem, copyCurrentVideoFrame, isVideoFrameCopyAvailable, openVideoDetails]
  );
  const isFilmstripVisible = getPreviewFilmstripVisible(previewValues);

  // Drop-to-compare: any all-image gallery-item drag dropped on the frame's drop zone
  // arms that image for comparison. The drag payload only carries names, so
  // the full contract is fetched before dispatching.
  const handleCompareDrop = useCallback(
    (event: DragEndEvent) => {
      if (selectedItem?.kind !== 'image') {
        return;
      }

      // The image on screen is refused, so this can never resolve to the
      // selection itself.
      const resolution = resolvePreviewCompareDrop(
        event.active.data.current,
        event.over?.data.current ?? null,
        selectedItem.name
      );

      if (!resolution) {
        return;
      }

      // Prefer images we already hold (board context includes fresh local
      // generations that would 404 on a backend by-name fetch).
      const localImageItem = boardItems.find((item) => item.kind === 'image' && item.name === resolution.imageName);

      if (localImageItem?.kind === 'image') {
        gallery.setCompareItem(localImageItem);
        return;
      }

      galleryImages
        .resolve(resolution.imageName)
        .then((image) => gallery.setCompareImage(image))
        .catch((error: unknown) => {
          notifications.reportError({
            area: 'preview-compare-drop',
            message: error instanceof Error ? error.message : String(error),
            namespace: 'gallery',
          });
        });
    },
    [boardItems, gallery, notifications, selectedItem]
  );
  useDndMonitor({ onDragEnd: handleCompareDrop });
  const openItemContextMenu = useCallback(
    (x: number, y: number) => {
      if (contextMenuItem) {
        setContextMenuTarget({
          itemRefs: [toGalleryItemRef(contextMenuItem)],
          items: [contextMenuItem],
          x,
          y,
        });
      }
    },
    [contextMenuItem]
  );
  const selectNextItem = useCallback(() => navigate(1), [navigate]);
  const selectPreviousItem = useCallback(() => navigate(-1), [navigate]);
  const closeContextMenu = useCallback(() => setContextMenuTarget(null), []);
  const headerItemName = shouldFollowLive ? null : (selectedItem?.name ?? null);

  // Publish the header chrome context (the "[board] / [image]" label and the
  // action strip's image + actions) for the widget frame; the chrome renders
  // outside this view, so an external store is the sync channel. Cleared on
  // unmount so stale chrome never outlives us.
  useEffect(() => {
    previewHeaderStore.set({
      actionItem: shouldFollowLive ? null : contextMenuItem,
      actions: shouldFollowLive ? null : imageActions,
      boardName: headerItemName === null ? null : boardName,
      copyCurrentVideoFrame: !shouldFollowLive && contextMenuItem?.kind === 'video' ? copyCurrentVideoFrame : null,
      isVideoFrameCopyAvailable: !shouldFollowLive && isVideoFrameCopyAvailable,
      itemName: headerItemName,
      openItemMenu: shouldFollowLive ? null : openItemContextMenu,
      openVideoDetails: !shouldFollowLive && contextMenuItem?.kind === 'video' ? openVideoDetails : null,
    });
  }, [
    boardName,
    contextMenuItem,
    copyCurrentVideoFrame,
    headerItemName,
    imageActions,
    isVideoFrameCopyAvailable,
    openItemContextMenu,
    openVideoDetails,
    shouldFollowLive,
  ]);

  useEffect(() => () => previewHeaderStore.clear(), []);

  const executeViewerHotkey = useEffectEvent((commandId: string) => {
    if (commandId === 'viewer.toggleViewer') {
      runtime.workbench.closeWidgetInstance(runtime.instanceId);
      return;
    }

    if (commandId === 'viewer.swapImages' && selectedItem?.kind === 'image' && compareImage) {
      selectPreviewItem(legacyGeneratedImageToGalleryItem(compareImage));
      gallery.setCompareItem(selectedItem);
      return;
    }

    if (commandId === 'viewer.deleteImage' && selectedItem && !shouldFollowLive) {
      void imageActions.deleteItems([toGalleryItemRef(selectedItem)]);
      return;
    }

    if (commandId === 'viewer.zoomToActual' && selectedItem?.kind === 'image') {
      loupeControlsRef.current?.zoomToActual();
      return;
    }

    if (commandId === 'viewer.zoomToFit' && selectedItem?.kind === 'image') {
      loupeControlsRef.current?.reset();
      return;
    }

    if (commandId === 'viewer.toggleFilmstrip') {
      widgets.patchValues('preview', { filmstripVisible: !getPreviewFilmstripVisible(previewValues) });
    }
  });

  useEffect(() => {
    const hotkeys = [
      ['viewer.toggleViewer', t('widgets.preview.commands.togglePreview'), ['z']],
      ['viewer.deleteImage', t('widgets.preview.commands.deletePreviewImage'), ['delete', 'backspace']],
      ['viewer.toggleFilmstrip', t('widgets.preview.commands.toggleFilmstrip'), ['t']],
      ...(selectedItem?.kind === 'image'
        ? ([
            ['viewer.swapImages', t('widgets.preview.commands.swapComparisonImages'), ['c']],
            ['viewer.zoomToActual', t('widgets.preview.commands.zoomToActual'), ['1']],
            ['viewer.zoomToFit', t('widgets.preview.commands.zoomToFit'), ['f']],
          ] as const)
        : []),
    ] as const;
    const disposers = hotkeys.flatMap(([id, title, defaultKeys]) => [
      runtime.commands.register({ handler: () => executeViewerHotkey(id), id, title }),
      runtime.hotkeys.register({ commandId: id, defaultKeys: [...defaultKeys], id, title }),
    ]);

    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [runtime.commands, runtime.hotkeys, selectedItem?.kind, t]);

  return (
    // No padding and no inner card: the dot-grid surface is the widget floor
    // and runs to every edge. `containerType` anchors the details panel's
    // `cqh` cap to the widget rather than the viewport.
    <Box ref={rootRef} containerType="size" h="full" position="relative" w="full">
      {/* Single always-mounted keyboard boundary: DOM focus survives swaps
          between the live, selected, and compare branches, so arrow
          navigation keeps working across them. */}
      <Stack
        aria-label={t('widgets.labels.preview')}
        gap="0"
        h="full"
        minH="0"
        outline="none"
        role="region"
        tabIndex={0}
        w="full"
        onKeyDown={handleNavigationKeyDown}
      >
        {shouldFollowLive && liveGalleryPlaceholders.length > 1 ? (
          <LivePreviewTiles
            placeholders={liveGalleryPlaceholders}
            shouldAntialiasProgressImage={antialiasProgressImages}
          />
        ) : shouldFollowLive && activeGalleryPlaceholder ? (
          <LivePreview
            boardItemCount={navigationSequence.length}
            density={density}
            filmstripItems={isFilmstripVisible && density !== 'minimal' ? boardItems : null}
            isLoadingBoard={isLoadingBoard}
            placeholder={activeGalleryPlaceholder}
            progressImage={matchingProgressImage}
            selectedIndex={navigationCursor}
            shouldAntialiasProgressImage={antialiasProgressImages}
            onNext={selectNextItem}
            onPrevious={selectPreviousItem}
            onSelectItem={selectPreviewItem}
          />
        ) : selectedItem ? (
          <>
            {isComparing && compareImage && selectedItem.kind === 'image' ? (
              <PreviewCompare
                baseImage={galleryImageItemToGalleryImage(selectedItem)}
                compareImage={compareImage}
                mode={comparisonMode}
                runtime={runtime}
                onExit={exitCompare}
                onModeChange={setComparisonMode}
                onSwap={swapCompareImages}
              />
            ) : selectedItem.kind === 'image' ? (
              <SelectedImagePreview
                actionImage={actionImage}
                actions={imageActions}
                boardItemCount={navigationSequence.length}
                density={density}
                filmstripItems={isFilmstripVisible && density !== 'minimal' ? boardItems : null}
                isItemCurrent={isItemCurrent}
                isLoadingBoard={isLoadingBoard}
                isMetadataOpen={isMetadataOpen}
                item={selectedItem}
                loupeControlsRef={loupeControlsRef}
                selectedIndex={navigationCursor}
                onContextMenu={openItemContextMenu}
                onNext={selectNextItem}
                onPrevious={selectPreviousItem}
                onSelectItem={selectPreviewItem}
                onToggleMetadata={toggleMetadata}
              />
            ) : (
              <SelectedVideoPreview
                actionImage={null}
                actions={imageActions}
                boardItemCount={navigationSequence.length}
                density={density}
                filmstripItems={isFilmstripVisible && density !== 'minimal' ? boardItems : null}
                isItemCurrent={isItemCurrent}
                isLoadingBoard={isLoadingBoard}
                isMetadataOpen={isMetadataOpen}
                item={selectedItem}
                videoControllerRef={videoControllerRef}
                selectedIndex={navigationCursor}
                onContextMenu={openItemContextMenu}
                onCopyAvailabilityChange={handleVideoCopyAvailabilityChange}
                onNext={selectNextItem}
                onPrevious={selectPreviousItem}
                onSelectItem={selectPreviewItem}
                onToggleMetadata={toggleMetadata}
              />
            )}
            <ImageContextMenu
              actions={imageActions}
              boards={boards}
              previewVideoActions={previewVideoContextActions}
              target={contextMenuTarget}
              onClose={closeContextMenu}
            />
            {deletionConfirmationDialog}
          </>
        ) : (
          <EmptyPreview />
        )}
      </Stack>
    </Box>
  );
};

const SelectedImagePreview = ({ item, ...props }: SelectedMediaPreviewProps & { item: GalleryImageItem }) => {
  const previewImage = useStreamingImageSource({
    fallbackImage: imageUrlToStreamingSource({
      alt: item.name,
      height: item.height,
      kind: 'fallback',
      src: item.fullUrl,
      width: item.width,
    }),
  });
  const source = useMemo<PreviewMediaSource | null>(
    () => (previewImage ? { itemKey: toGalleryItemKey(item), kind: 'image', source: previewImage } : null),
    [item, previewImage]
  );

  return (
    <SelectedMediaPreview
      {...props}
      dragItem={toGalleryItemRef(item)}
      frameHeight={previewImage?.height ?? item.height}
      frameWidth={previewImage?.width ?? item.width}
      item={item}
      source={source}
    />
  );
};

const SelectedVideoPreview = ({
  item,
  ...props
}: SelectedMediaPreviewProps & { item: Extract<GalleryItem, { kind: 'video' }> }) => {
  const { t } = useTranslation();
  const source = useMemo<PreviewMediaSource>(
    () => ({
      itemKey: toGalleryItemKey(item),
      kind: 'video',
      label: t('widgets.preview.videoLabel', { name: item.name }),
      poster: item.thumbnailUrl,
      src: item.fullUrl,
    }),
    [item, t]
  );

  return (
    <SelectedMediaPreview {...props} frameHeight={item.height} frameWidth={item.width} item={item} source={source} />
  );
};

interface SelectedMediaPreviewProps {
  actionImage: GalleryImage | null;
  actions: ImageActions;
  boardItemCount: number;
  density: PreviewDensity;
  /** Board thumbnails for the filmstrip, or null when the strip is hidden. */
  filmstripItems: GalleryItem[] | null;
  isItemCurrent: (itemKey: GalleryItemKey) => boolean;
  isLoadingBoard: boolean;
  isMetadataOpen: boolean;
  item: GalleryItem;
  loupeControlsRef?: Ref<PreviewLoupeControls>;
  onCopyAvailabilityChange?: (itemKey: GalleryItemKey, isAvailable: boolean) => void;
  selectedIndex: number;
  onContextMenu: (x: number, y: number) => void;
  onNext: () => void;
  onPrevious: () => void;
  onSelectItem: (item: GalleryItem) => void;
  onToggleMetadata: () => void;
  videoControllerRef?: Ref<PreviewVideoFrameController>;
}

/**
 * The one media arrangement, shared by selected items and the live preview:
 * the stage fills, and the filmstrip + footer float above its lower edge in a
 * single island stack. The stage reserves bottom padding for the footer island
 * only, so the fitted media is never tucked under it; the filmstrip floats
 * over the media itself and steals no height. Live and finished renders MUST
 * pass through the same scaffold — the denoise→done boundary may change only
 * the pixels inside the frame, never the geometry around it.
 */
const PreviewMediaScaffold = ({ children }: { children: ReactNode }) => (
  <Flex direction="column" h="full" minH="0" position="relative" w="full">
    {children}
  </Flex>
);

/** The floating island column the filmstrip and footer share. */
const PreviewOverlayStack = ({ children }: { children: ReactNode }) => (
  <Stack bottom="2" gap="2" insetX="2" position="absolute" zIndex="1">
    {children}
  </Stack>
);

const getMediaStagePadding = (density: PreviewDensity): string => (density === 'full' ? '6' : '3');

const SelectedMediaPreview = ({
  actionImage,
  actions,
  boardItemCount,
  density,
  dragItem,
  filmstripItems,
  frameHeight,
  frameWidth,
  isItemCurrent,
  isLoadingBoard,
  isMetadataOpen,
  item,
  loupeControlsRef,
  onCopyAvailabilityChange,
  selectedIndex,
  source,
  onContextMenu,
  onNext,
  onPrevious,
  onSelectItem,
  onToggleMetadata,
  videoControllerRef,
}: SelectedMediaPreviewProps & {
  dragItem?: GalleryItemRef;
  frameHeight: number;
  frameWidth: number;
  source: Parameters<typeof PreviewFrame>[0]['source'];
}) => {
  const media = useMemo<PreviewFooterMedia>(
    () => ({ actionImage, actions, item, kind: 'item' }),
    [actionImage, actions, item]
  );

  return (
    <PreviewMediaScaffold>
      <PreviewFrame
        dragItem={dragItem}
        frameHeight={frameHeight}
        frameWidth={frameWidth}
        isItemCurrent={isItemCurrent}
        isLive={false}
        loupeControlsRef={loupeControlsRef}
        onVideoCopyAvailabilityChange={onCopyAvailabilityChange}
        padding={getMediaStagePadding(density)}
        paddingBottom={PREVIEW_OVERLAY_RESERVE}
        shouldAntialiasLiveImage
        source={source}
        variant="framed"
        videoControllerRef={videoControllerRef}
        onContextMenu={onContextMenu}
      />
      <PreviewOverlayStack>
        {filmstripItems ? (
          <PreviewFilmstrip
            density={density}
            items={filmstripItems}
            selectedItemKey={toGalleryItemKey(item)}
            onSelect={onSelectItem}
          />
        ) : null}
        <PreviewFooter
          boardItemCount={boardItemCount}
          isLoadingBoard={isLoadingBoard}
          isMetadataOpen={isMetadataOpen}
          media={media}
          selectedIndex={selectedIndex}
          onNext={onNext}
          onPrevious={onPrevious}
          onToggleMetadata={onToggleMetadata}
        />
      </PreviewOverlayStack>
    </PreviewMediaScaffold>
  );
};

/**
 * The single-session live preview: the denoise stream rendered exactly like a
 * finished item — same scaffold, same frame chrome, no badge — so the moment
 * generation completes, only the pixels change. The footer stays up
 * throughout, fed by queue data: the slot's position in the same navigation
 * sequence the arrow keys walk, and its requested output size.
 */
const LivePreview = ({
  boardItemCount,
  density,
  filmstripItems,
  isLoadingBoard,
  placeholder,
  progressImage,
  selectedIndex,
  shouldAntialiasProgressImage,
  onNext,
  onPrevious,
  onSelectItem,
}: {
  boardItemCount: number;
  density: PreviewDensity;
  filmstripItems: GalleryItem[] | null;
  isLoadingBoard: boolean;
  placeholder: GalleryQueuePlaceholder;
  progressImage: LatestProgressImageSnapshot | null;
  selectedIndex: number;
  shouldAntialiasProgressImage: boolean;
  onNext: () => void;
  onPrevious: () => void;
  onSelectItem: (item: GalleryItem) => void;
}) => {
  const previewImage = useStreamingImageSource({
    liveImage: progressImageToStreamingSource(progressImage),
  });
  const source = useMemo<PreviewMediaSource | null>(
    () =>
      previewImage
        ? {
            itemKey: `image:live:${placeholder.id}`,
            kind: 'image',
            source: previewImage,
          }
        : null,
    [placeholder.id, previewImage]
  );

  const media = useMemo<PreviewFooterMedia>(
    () => ({ height: placeholder.height, kind: 'live', width: placeholder.width }),
    [placeholder.height, placeholder.width]
  );

  return (
    <PreviewMediaScaffold>
      <PreviewFrame
        frameHeight={previewImage?.height ?? placeholder.height}
        frameWidth={previewImage?.width ?? placeholder.width}
        isLive
        padding={getMediaStagePadding(density)}
        paddingBottom={PREVIEW_OVERLAY_RESERVE}
        shouldAntialiasLiveImage={shouldAntialiasProgressImage}
        source={source}
        variant="framed"
      />
      <PreviewOverlayStack>
        {filmstripItems ? (
          <PreviewFilmstrip density={density} items={filmstripItems} selectedItemKey={null} onSelect={onSelectItem} />
        ) : null}
        <PreviewFooter
          boardItemCount={boardItemCount}
          isLoadingBoard={isLoadingBoard}
          isMetadataOpen={false}
          media={media}
          selectedIndex={selectedIndex}
          onNext={onNext}
          onPrevious={onPrevious}
          onToggleMetadata={noop}
        />
      </PreviewOverlayStack>
    </PreviewMediaScaffold>
  );
};

/**
 * One tile in the multi-session grid.
 *
 * Subscribes to its own slot's progress image rather than receiving it from the
 * parent, so a frame from one GPU's session re-renders only that tile.
 */
export const LivePreviewTile = ({
  placeholder,
  shouldAntialiasProgressImage,
}: {
  placeholder: GalleryQueuePlaceholder;
  shouldAntialiasProgressImage: boolean;
}) => {
  const { t } = useTranslation();
  const progressImage = useQueueItemProgressImage(placeholder.queueItemId, placeholder.itemIndex);
  // Keyed by the backend item id, not the local one: two slots of the same batch can
  // be running on two GPUs, and the local-keyed store holds one entry for both.
  const itemProgress = useItemProgress(placeholder.backendItemId);
  const deviceLabel = useDeviceLabel(itemProgress?.device);
  const previewImage = useStreamingImageSource({
    liveImage: progressImageToStreamingSource(progressImage),
  });
  const source = useMemo<PreviewMediaSource | null>(
    () =>
      previewImage
        ? {
            itemKey: `image:live:${placeholder.id}`,
            kind: 'image',
            source: previewImage,
          }
        : null,
    [placeholder.id, previewImage]
  );

  const percent = getDeterminateProgressPercent(itemProgress?.percentage);
  // Every tile declares itself live: with several sessions racing, a silent
  // tile reads as a stuck one. Device label when known, plain "Generating"
  // otherwise; percent appended once quantified (zero is model-loading).
  const badgeBase = deviceLabel
    ? t('widgets.queue.device.shortLabel', { index: deviceLabel.index })
    : t('common.generating');
  const liveBadgeLabel = percent === null ? badgeBase : `${badgeBase} · ${percent}%`;

  return (
    <PreviewFrame
      frameHeight={previewImage?.height ?? placeholder.height}
      frameWidth={previewImage?.width ?? placeholder.width}
      isLive
      liveBadgeLabel={liveBadgeLabel}
      shouldAntialiasLiveImage={shouldAntialiasProgressImage}
      source={source}
      variant="inset"
    />
  );
};

/**
 * Side-by-side previews for concurrent sessions (multi-GPU).
 *
 * Only mounted for two or more live slots; a single session keeps the full-size
 * single-frame preview so nothing changes on a single-GPU install. The grid is a
 * plain auto-fit so two GPUs sit side by side and four wrap to a 2×2.
 */
export const LivePreviewTiles = ({
  placeholders,
  shouldAntialiasProgressImage,
}: {
  placeholders: GalleryQueuePlaceholder[];
  shouldAntialiasProgressImage: boolean;
}) => (
  <SimpleGrid gap="2" h="full" minH="0" columns={placeholders.length > 2 ? 2 : placeholders.length}>
    {placeholders.map((placeholder) => (
      <LivePreviewTile
        key={placeholder.id}
        placeholder={placeholder}
        shouldAntialiasProgressImage={shouldAntialiasProgressImage}
      />
    ))}
  </SimpleGrid>
);

const EmptyPreview = () => {
  const { t } = useTranslation();

  return (
    <PreviewFrame frameHeight={1} frameWidth={1} isLive={false} shouldAntialiasLiveImage source={null} variant="inset">
      <Stack align="center" color="fg" gap="2" maxW="18rem" textAlign="center">
        <Text fontSize="sm" fontWeight="800">
          {t('widgets.preview.noGallerySelection')}
        </Text>
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.preview.emptyDescription')}
        </Text>
      </Stack>
    </PreviewFrame>
  );
};

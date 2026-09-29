import type { SystemStyleObject } from '@chakra-ui/react';
import type { QueueActiveSession, QueueItem } from '@features/queue/contracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Box, Flex, Stack, Text } from '@chakra-ui/react';
import { useDndMonitor, type DragEndEvent } from '@dnd-kit/core';
import {
  galleryImages,
  type GalleryBoard,
  type GalleryImageItem,
  type GalleryItem,
  type GalleryItemKey,
  type GalleryItemRef,
} from '@features/gallery';
import {
  getGalleryCompareImage,
  getGalleryPage,
  getGallerySelectedImageQuery,
  getGallerySemanticImageQuery,
  getGallerySelectedBoardId,
  getGallerySettings,
  getSelectedGalleryItemFromValues,
  getBoundedRecentImages,
  galleryImageItemToGalleryImage,
  isGalleryImageItem,
  legacyGeneratedImageToGalleryItem,
  normalizeGalleryImage,
  requestGalleryItemReveal,
  toGalleryItemKey,
  toGalleryItemRef,
} from '@features/gallery/contracts';
import { galleryBoardsOptions, getGalleryListingBoardsQuery } from '@features/gallery/queries';
import { createGenerateFormValuesSelector } from '@features/generation/react';
import {
  consumeQueueItemSwapProgressImage,
  useQueueItemBridgeProgressImage,
  useQueueItemProgressImage,
  useQueueItemSwapProgressImage,
} from '@features/queue/react';
import {
  imageUrlToStreamingSource,
  progressImageToStreamingSource,
  type StreamingImageSource,
} from '@platform/ui/streaming-image/streamingImageSource';
import { useStreamingImageSource } from '@platform/ui/streaming-image/useStreamingImageSource';
import { useQuery } from '@tanstack/react-query';
import {
  ImageContextMenu,
  useDeletionConfirmation,
  useImageActions,
  type ImageContextMenuTarget,
} from '@workbench/image-actions';
import { QueueProgressRail } from '@workbench/queue-integration/QueueProgressRail';
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
import type { PreviewSwipeNavigation } from './usePreviewSwipe';

import { useLivePreviewFollow } from './livePreviewFollow';
import { PreviewCompare } from './PreviewCompare';
import { resolvePreviewCompareDrop } from './previewCompareDnd';
import { usePreviewDensity, type PreviewDensity } from './previewDensity';
import { PreviewFilmstrip } from './PreviewFilmstrip';
import {
  PreviewFrame,
  type PreviewMediaSource,
  type PreviewVideoFrameController,
  type PreviewVideoFrameCopyResult,
} from './PreviewFrame';
import { previewHeaderStore, previewStageStore, type PreviewZoomCommands } from './previewHeaderStore';
import { getPreviewComparisonMode, getPreviewFilmstripVisible, type PreviewComparisonMode } from './previewSettings';
import { usePreviewNavigation } from './usePreviewNavigation';

/** For the live footer's Details slot, which renders disabled and never fires. */
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

const EMPTY_BOARDS: GalleryBoard[] = [];

const fallbackBoards: GalleryBoard[] = [
  {
    archived: false,
    assetCount: 0,
    assetVideoCount: 0,
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

const selectGenerateRecallValues = createGenerateFormValuesSelector();

/** Pinned to the floating window body's top edge, under the title bar's divider. */
const FLOATING_RAIL_SX: SystemStyleObject = {
  display: 'flex',
  gap: '1px',
  height: '3px',
  insetInline: 0,
  pointerEvents: 'none',
  position: 'absolute',
  top: 0,
  zIndex: 3,
};

export const PreviewWidgetView = ({ region, runtime }: WidgetViewProps) => {
  const galleryValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'gallery'));
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const previewValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'preview'));
  const generateValues = useWidgetValuesSelector('generate', selectGenerateRecallValues);
  const { antialiasProgressImages } = useActiveProjectSelector((project) => project.settings);
  const livePreview = useLivePreviewFollow();
  const { gallery, notifications, widgets } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const { density, rootRef } = usePreviewDensity(region);
  const recentImages = galleryValues.recentImages;
  const localItems = useMemo(() => getLocalGalleryItems({ recentImages }, queueItems), [queueItems, recentImages]);
  const selectedItem = useMemo(() => getSelectedItem(galleryValues, localItems), [galleryValues, localItems]);
  const compareImage = getGalleryCompareImage(galleryValues);
  const comparisonMode = getPreviewComparisonMode(previewValues);
  const displayBoardId = selectedItem?.boardId ?? 'none';
  const hasSelectedItem = selectedItem !== null;
  const selectedImageQuery = getGallerySelectedImageQuery(galleryValues);
  // Memoize parsed ranking by raw value so unrelated Gallery updates do not rebuild navigation.
  const gallerySemanticQuery = useMemo(
    () => getGallerySemanticImageQuery({ semanticImageQuery: galleryValues.semanticImageQuery }),
    [galleryValues.semanticImageQuery]
  );
  const selectedItemKey = selectedItem ? toGalleryItemKey(selectedItem) : null;
  const activeGalleryPlaceholder =
    livePreview.sessions.find((session) => session.id === livePreview.followedSessionId) ?? null;
  const shouldFollowLive = activeGalleryPlaceholder !== null;
  const isComparing =
    !shouldFollowLive &&
    selectedItem?.kind === 'image' &&
    compareImage !== null &&
    toGalleryItemKey({ kind: 'image', name: compareImage.imageName }) !== selectedItemKey;
  const { t } = useTranslation();
  const navigationBoundaryRef = useRef<HTMLDivElement | null>(null);
  const loupeControlsRef = useRef<PreviewLoupeControls | null>(null);
  // The header's zoom commands close over the controls ref, so they are stable
  // per mount; the readout itself streams through `previewStageStore`.
  const videoControllerRef = useRef<PreviewVideoFrameController | null>(null);
  const zoomCommands = useMemo<PreviewZoomCommands>(
    () => ({
      reset: () => loupeControlsRef.current?.reset(),
      zoomTo: (actualZoom: number) => loupeControlsRef.current?.zoomTo(actualZoom),
    }),
    []
  );
  const [copyAvailableItemKey, setCopyAvailableItemKey] = useState<GalleryItemKey | null>(null);

  const boardsQuery = useQuery({
    ...galleryBoardsOptions(),
    enabled: hasSelectedItem,
  });
  // A ranked filmstrip mirrors the grid, which ranks within the board it resolves; share the grid's board list.
  const gallerySettings = getGallerySettings(galleryValues);
  const galleryBoardsQuery = useQuery({
    ...galleryBoardsOptions(getGalleryListingBoardsQuery(gallerySettings)),
    enabled: gallerySemanticQuery !== null,
  });
  const galleryBoardId = getGallerySelectedBoardId(galleryValues, galleryBoardsQuery.data ?? EMPTY_BOARDS);
  const boards = boardsQuery.data ?? fallbackBoards;
  const boardName = getBoardName(
    boards,
    displayBoardId,
    t('widgets.gallery.uncategorized'),
    t('widgets.gallery.unknownBoard')
  );

  const selectGalleryItemAtPage = useCallback(
    (item: GalleryItem, selectionPage: number) => {
      gallery.selectItem(item, undefined, selectionPage, true);
      // Deliberate navigation: the grid follows it, unlike auto-selection.
      requestGalleryItemReveal(toGalleryItemKey(item));
    },
    [gallery]
  );
  const {
    boardItems,
    getSelectionPage,
    handleNavigationKeyDown,
    isLoadingBoard,
    navigate,
    navigationCursor,
    navigationQueryKey,
    neighbors,
    selectPreviewItem,
  } = usePreviewNavigation({
    followedSessionId: activeGalleryPlaceholder?.id ?? null,
    followSession: livePreview.follow,
    isComparing,
    localItems,
    progressSessions: livePreview.gallerySessions,
    queueItems,
    selectGalleryItem: selectGalleryItemAtPage,
    selectedImageQuery,
    selectedItem,
    galleryBoardId,
    galleryPage: getGalleryPage(galleryValues),
    galleryPaginationMode: gallerySettings.paginationMode,
    selectedItemKey,
    semanticQuery: gallerySemanticQuery,
  });

  const swipeNavigation = useMemo<PreviewSwipeNavigation>(
    () => ({ neighbors, onNavigate: navigate }),
    [navigate, neighbors]
  );

  const [contextMenuTarget, setContextMenuTarget] = useState<ImageContextMenuTarget | null>(null);
  const getItemActionContext = useCallback(
    () => ({
      filterIdentity: navigationQueryKey,
      getItemSelectionPage: getSelectionPage,
      items: boardItems,
      loadOrderedRefs: (signal: AbortSignal) => {
        signal.throwIfAborted();
        return Promise.resolve(boardItems.map(toGalleryItemRef));
      },
      selectedItemKey,
    }),
    [boardItems, getSelectionPage, navigationQueryKey, selectedItemKey]
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
  const exitCompare = useCallback(() => gallery.setCompareItem(null), [gallery]);
  // Remember the comparison item's original page for swap-back, but only within its original query and board. Item
  // moves can invalidate the page without changing query identity.
  const swappedOutRef = useRef<{ boardId: string; key: GalleryItemKey; page: number; queryKey: string } | null>(null);
  const swapCompareImages = useCallback(() => {
    if (selectedItem?.kind === 'image' && compareImage) {
      const compareItem = legacyGeneratedImageToGalleryItem(compareImage);
      const swappedOut = swappedOutRef.current;

      swappedOutRef.current = {
        boardId: selectedItem.boardId,
        key: toGalleryItemKey(selectedItem),
        page: selectedImageQuery.page,
        queryKey: navigationQueryKey,
      };
      if (
        swappedOut &&
        swappedOut.key === toGalleryItemKey(compareItem) &&
        swappedOut.queryKey === navigationQueryKey &&
        swappedOut.boardId === compareItem.boardId
      ) {
        selectGalleryItemAtPage(compareItem, swappedOut.page);
      } else {
        selectPreviewItem(compareItem);
      }
      gallery.setCompareItem(selectedItem);
    }
  }, [
    compareImage,
    gallery,
    navigationQueryKey,
    selectGalleryItemAtPage,
    selectPreviewItem,
    selectedImageQuery.page,
    selectedItem,
  ]);
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

  // Hydrate dropped image names before arming comparison with a complete item contract.
  const handleCompareDrop = useCallback(
    (event: DragEndEvent) => {
      if (selectedItem?.kind !== 'image') {
        return;
      }

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
  const openFilmstripItemContextMenu = useCallback(
    (item: GalleryItem, x: number, y: number) =>
      setContextMenuTarget({ itemRefs: [toGalleryItemRef(item)], items: [item], x, y }),
    []
  );
  const compareFilmstripItem = useCallback(
    (item: GalleryItem) => {
      if (isGalleryImageItem(item)) {
        imageActions.selectForCompare(galleryImageItemToGalleryImage(item));
      }
    },
    [imageActions]
  );
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
  const closeContextMenu = useCallback(() => setContextMenuTarget(null), []);
  const headerItemName = shouldFollowLive ? null : (selectedItem?.name ?? null);

  // Publish selection/actions to hoisted chrome through an external store; clear on unmount to prevent stale
  // headers.
  const filmstrip = useMemo<PreviewFilmstripProps | null>(
    () =>
      isFilmstripVisible && density !== 'minimal'
        ? {
            followedSessionId: livePreview.followedSessionId,
            isSessionPinned: livePreview.pinnedSessionId !== null,
            items: boardItems,
            onCompare: compareFilmstripItem,
            onContextMenu: openFilmstripItemContextMenu,
            onFollowSession: livePreview.follow,
            onSelect: selectPreviewItem,
            onUnpinSession: livePreview.showAll,
            sessions: livePreview.gallerySessions,
            shouldAntialiasLiveImage: antialiasProgressImages,
          }
        : null,
    [
      antialiasProgressImages,
      boardItems,
      compareFilmstripItem,
      density,
      isFilmstripVisible,
      livePreview.follow,
      livePreview.followedSessionId,
      livePreview.gallerySessions,
      livePreview.pinnedSessionId,
      livePreview.showAll,
      openFilmstripItemContextMenu,
      selectPreviewItem,
    ]
  );
  const hasHeaderItem = !shouldFollowLive && contextMenuItem !== null;
  useEffect(() => {
    previewHeaderStore.set({
      actionItem: shouldFollowLive ? null : contextMenuItem,
      actions: shouldFollowLive ? null : imageActions,
      boardName: headerItemName === null ? null : boardName,
      copyCurrentVideoFrame: !shouldFollowLive && contextMenuItem?.kind === 'video' ? copyCurrentVideoFrame : null,
      isVideoFrameCopyAvailable: !shouldFollowLive && isVideoFrameCopyAvailable,
      itemName: headerItemName,
      openItemMenu: shouldFollowLive ? null : openItemContextMenu,
      position: hasHeaderItem
        ? { boardItemCount: boardItems.length, isLoadingBoard, selectedIndex: navigationCursor }
        : null,
      // Only the single image frame carries a loupe: videos have none and
      // compare has its own synced pair.
      zoom: hasHeaderItem && !isComparing && contextMenuItem.kind === 'image' ? zoomCommands : null,
    });
  }, [
    boardItems.length,
    boardName,
    contextMenuItem,
    copyCurrentVideoFrame,
    hasHeaderItem,
    headerItemName,
    imageActions,
    isComparing,
    isLoadingBoard,
    isVideoFrameCopyAvailable,
    navigationCursor,
    openItemContextMenu,
    shouldFollowLive,
    zoomCommands,
  ]);

  useEffect(
    () => () => {
      previewHeaderStore.clear();
      previewStageStore.clear();
    },
    []
  );

  const executeViewerHotkey = useEffectEvent((commandId: string) => {
    if (commandId === 'viewer.swapImages' && selectedItem?.kind === 'image' && compareImage) {
      swapCompareImages();
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
    // Let the dot grid fill the widget; containerType bounds Details cqh sizing to this surface.
    <Box ref={rootRef} containerType="size" h="full" position="relative" w="full">
      {/*
       * Show progress beneath floating Preview's title bar when the main topbar may be out of view; overlay it
       * without reflow.
       */}
      {region === 'floating' ? <QueueProgressRail css={FLOATING_RAIL_SX} /> : null}
      {/* Single always-mounted keyboard boundary: DOM focus survives swaps
          between the live, selected, and compare branches, so arrow
          navigation keeps working across them. */}
      <Stack
        ref={navigationBoundaryRef}
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
        {shouldFollowLive && activeGalleryPlaceholder ? (
          <LivePreview
            density={density}
            filmstrip={filmstrip}
            placeholder={activeGalleryPlaceholder}
            shouldAntialiasProgressImage={antialiasProgressImages}
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
                density={density}
                filmstrip={filmstrip}
                isItemCurrent={isItemCurrent}
                item={selectedItem}
                loupeControlsRef={loupeControlsRef}
                shouldAntialiasProgressImage={antialiasProgressImages}
                swipe={swipeNavigation}
                onContextMenu={openItemContextMenu}
              />
            ) : (
              <SelectedVideoPreview
                density={density}
                filmstrip={filmstrip}
                isItemCurrent={isItemCurrent}
                item={selectedItem}
                swipe={swipeNavigation}
                videoControllerRef={videoControllerRef}
                onContextMenu={openItemContextMenu}
                onCopyAvailabilityChange={handleVideoCopyAvailabilityChange}
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

const SelectedImagePreview = ({
  item,
  shouldAntialiasProgressImage,
  ...props
}: SelectedMediaPreviewProps & { item: GalleryImageItem; shouldAntialiasProgressImage: boolean }) => {
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
  // Bridge finished-image decoding with its run's final denoise frame so only pixels change at completion.
  const swapProgressImage = useQueueItemSwapProgressImage(item.sourceQueueItemId, item.name);
  const holdSource = useMemo(
    () => progressImageToStreamingSource(swapProgressImage, item.name),
    [item.name, swapProgressImage]
  );
  const sourceQueueItemId = item.sourceQueueItemId;
  const handleSourceLoaded = useCallback(() => {
    if (sourceQueueItemId) {
      consumeQueueItemSwapProgressImage(sourceQueueItemId);
    }
  }, [sourceQueueItemId]);

  return (
    <SelectedMediaPreview
      {...props}
      dragItem={toGalleryItemRef(item)}
      frameHeight={previewImage?.height ?? item.height}
      frameWidth={previewImage?.width ?? item.width}
      holdSource={holdSource}
      item={item}
      placeholderSrc={item.thumbnailUrl}
      shouldAntialiasHoldImage={shouldAntialiasProgressImage}
      source={source}
      onSourceLoaded={handleSourceLoaded}
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
    <SelectedMediaPreview
      {...props}
      dragItem={toGalleryItemRef(item)}
      frameHeight={item.height}
      frameWidth={item.width}
      item={item}
      source={source}
    />
  );
};

/** The filmstrip's inputs, or null when the strip is hidden. */
type PreviewFilmstripProps = Omit<Parameters<typeof PreviewFilmstrip>[0], 'density' | 'selectedItemKey'>;

interface SelectedMediaPreviewProps {
  density: PreviewDensity;
  filmstrip: PreviewFilmstripProps | null;
  isItemCurrent: (itemKey: GalleryItemKey) => boolean;
  item: GalleryItem;
  loupeControlsRef?: Ref<PreviewLoupeControls>;
  onCopyAvailabilityChange?: (itemKey: GalleryItemKey, isAvailable: boolean) => void;
  onContextMenu: (x: number, y: number) => void;
  swipe?: PreviewSwipeNavigation;
  videoControllerRef?: Ref<PreviewVideoFrameController>;
}

/**
 * Use the same stage/filmstrip scaffold for live and finished media so completion changes pixels, not geometry;
 * multiple sessions remain filmstrip choices.
 */
const PreviewMediaScaffold = ({ children }: { children: ReactNode }) => (
  <Flex direction="column" h="full" minH="0" position="relative" w="full">
    {children}
  </Flex>
);

const getMediaStagePadding = (density: PreviewDensity): string => (density === 'full' ? '6' : '3');
const { setStageElement, setZoom: setZoomReadout } = previewStageStore;

const SelectedMediaPreview = ({
  density,
  dragItem,
  filmstrip,
  frameHeight,
  frameWidth,
  holdSource,
  isItemCurrent,
  item,
  loupeControlsRef,
  onCopyAvailabilityChange,
  onSourceLoaded,
  placeholderSrc,
  shouldAntialiasHoldImage,
  source,
  swipe,
  onContextMenu,
  videoControllerRef,
}: SelectedMediaPreviewProps & {
  dragItem?: GalleryItemRef;
  frameHeight: number;
  frameWidth: number;
  holdSource?: StreamingImageSource | null;
  onSourceLoaded?: (src: string) => void;
  placeholderSrc?: string;
  shouldAntialiasHoldImage?: boolean;
  source: Parameters<typeof PreviewFrame>[0]['source'];
}) => (
  <PreviewMediaScaffold>
    {/* The stage wrapper is the Details popover's boundary for images and
        videos alike; the frame inside fills it, so their rects coincide. */}
    <Flex ref={setStageElement} direction="column" flex="1" minH="0">
      <PreviewFrame
        dragItem={dragItem}
        frameHeight={frameHeight}
        frameWidth={frameWidth}
        holdSource={holdSource}
        isItemCurrent={isItemCurrent}
        isLive={false}
        loupeControlsRef={loupeControlsRef}
        onSourceLoaded={onSourceLoaded}
        onVideoCopyAvailabilityChange={onCopyAvailabilityChange}
        onZoomChange={setZoomReadout}
        padding={getMediaStagePadding(density)}
        placeholderSrc={placeholderSrc}
        shouldAntialiasLiveImage={shouldAntialiasHoldImage ?? true}
        source={source}
        swipe={swipe}
        variant="framed"
        videoControllerRef={videoControllerRef}
        onContextMenu={onContextMenu}
      />
    </Flex>
    {filmstrip ? <PreviewFilmstrip {...filmstrip} density={density} selectedItemKey={toGalleryItemKey(item)} /> : null}
  </PreviewMediaScaffold>
);

/** Match finished-frame geometry for live previews; saved-image navigation stays in the filmstrip. */
const LivePreview = ({
  density,
  filmstrip,
  placeholder,
  shouldAntialiasProgressImage,
}: {
  density: PreviewDensity;
  filmstrip: PreviewFilmstripProps | null;
  placeholder: QueueActiveSession;
  shouldAntialiasProgressImage: boolean;
}) => {
  // Use the followed slot's own frame so another slot's progress or release cannot replace it.
  const progressImage = useQueueItemProgressImage(placeholder.queueItemId, placeholder.itemIndex);
  // The previous slot's last frame stands in until this slot produces one of
  // its own (model load, text encoding) — otherwise a sequential batch drops
  // to an empty card between items.
  const bridgeProgressImage = useQueueItemBridgeProgressImage(placeholder.queueItemId);
  const previewImage = useStreamingImageSource({
    heldLiveImage: progressImageToStreamingSource(bridgeProgressImage),
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

  return (
    <PreviewMediaScaffold>
      <PreviewFrame
        frameHeight={previewImage?.height ?? placeholder.height}
        frameWidth={previewImage?.width ?? placeholder.width}
        isLive
        padding={getMediaStagePadding(density)}
        shouldAntialiasLiveImage={shouldAntialiasProgressImage}
        source={source}
        variant="framed"
      />
      {filmstrip ? <PreviewFilmstrip {...filmstrip} density={density} selectedItemKey={null} /> : null}
    </PreviewMediaScaffold>
  );
};

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

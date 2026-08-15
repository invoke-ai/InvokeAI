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
  type GalleryView,
} from '@features/gallery';
import {
  getGalleryCompareImage,
  getGalleryGenerationSequence,
  getGalleryLiveSlots,
  getGallerySelectedImageQuery,
  getGallerySettings,
  getSelectedGalleryItemFromValues,
  getBoundedRecentImages,
  compareGalleryItems,
  galleryImageItemToGalleryImage,
  isGalleryImageItem,
  legacyGeneratedImageToGalleryItem,
  normalizeGalleryImage,
  toGalleryItemKey,
  toGalleryItemRef,
  type GalleryItemsPage,
  type GalleryQueuePlaceholder,
} from '@features/gallery/contracts';
import {
  flattenGalleryItemsData,
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
  galleryBoardsOptions,
  galleryItemsInfiniteOptions,
} from '@features/gallery/queries';
import { createGenerateFormValuesSelector } from '@features/generation/react';
import { useDeviceLabel } from '@features/queue/devices';
import {
  useActiveProgressTarget,
  useActiveProgressTargets,
  useItemProgress,
  useProgressImage,
  useQueueItemProgressImage,
  type LatestProgressImageSnapshot,
} from '@features/queue/react';
import { parseDateTokens } from '@platform/search/dateTokens';
import {
  imageUrlToStreamingSource,
  progressImageToStreamingSource,
} from '@platform/ui/streaming-image/streamingImageSource';
import { useStreamingImageSource } from '@platform/ui/streaming-image/useStreamingImageSource';
import { useInfiniteQuery, useQuery, type InfiniteData } from '@tanstack/react-query';
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
import { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState, type KeyboardEvent, type Ref } from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewLoupeControls } from './usePreviewLoupe';

import { PreviewCompare } from './PreviewCompare';
import { resolvePreviewCompareDrop } from './previewCompareDnd';
import { usePreviewDensity, type PreviewDensity } from './previewDensity';
import { PreviewFilmstrip } from './PreviewFilmstrip';
import { PreviewFooter } from './PreviewFooter';
import {
  PreviewFrame,
  type PreviewMediaSource,
  type PreviewVideoFrameController,
  type PreviewVideoFrameCopyResult,
} from './PreviewFrame';
import { previewHeaderStore } from './previewHeaderStore';
import {
  getPreviewNavigationCursor,
  getPreviewNavigationSequence,
  getPreviewNavigationTarget,
} from './previewNavigation';
import {
  getPreviewComparisonMode,
  getPreviewFilmstripVisible,
  getPreviewMetadataOpen,
  type PreviewComparisonMode,
} from './previewSettings';

const EMPTY_PREVIEW_ITEMS: GalleryItem[] = [];

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

const flattenPreviewItems = (data: InfiniteData<GalleryItemsPage, number> | undefined): GalleryItem[] =>
  flattenGalleryItemsData(data);

const getOrderedPreviewItems = (
  items: GalleryItem[],
  imageOrderDir: 'ASC' | 'DESC',
  starredFirst: boolean,
  inputOrder: 'display' | 'newest-first'
): GalleryItem[] =>
  items
    .map((item, index) => ({ index, item }))
    .sort((a, b) => {
      const canonicalOrder = compareGalleryItems(a.item, b.item, { orderDir: imageOrderDir, starredFirst });

      if (canonicalOrder !== 0) {
        return canonicalOrder;
      }

      return inputOrder === 'newest-first' && imageOrderDir === 'ASC' ? b.index - a.index : a.index - b.index;
    })
    .map(({ item }) => item);

const getOrderedLocalItems = ({
  boardId,
  galleryView,
  items,
  imageOrderDir,
  starredFirst,
}: {
  boardId: string;
  galleryView: GalleryView;
  items: GalleryItem[];
  imageOrderDir: 'ASC' | 'DESC';
  starredFirst: boolean;
}): GalleryItem[] =>
  getOrderedPreviewItems(
    items.filter((item) => item.boardId === boardId && getItemGalleryView(item) === galleryView),
    imageOrderDir,
    starredFirst,
    'newest-first'
  );

export const mergePreviewBoardItems = (
  backendItems: GalleryItem[],
  localItems: GalleryItem[],
  imageOrderDir: 'ASC' | 'DESC',
  starredFirst: boolean
): GalleryItem[] => {
  const backendKeys = new Set(backendItems.map(toGalleryItemKey));
  const missingLocalItems = localItems.filter((item) => !backendKeys.has(toGalleryItemKey(item)));

  if (missingLocalItems.length === 0) {
    return backendItems.slice(0, GALLERY_MAX_ROWS);
  }

  return getOrderedPreviewItems([...backendItems, ...missingLocalItems], imageOrderDir, starredFirst, 'display').slice(
    0,
    GALLERY_MAX_ROWS
  );
};

/**
 * The board used for the "N of M" index comes from the image itself, never
 * from the gallery's currently selected board — switching boards or changing
 * the board list sort must not reshuffle the preview. Freshly generated local
 * images (no boardId yet) fall back to the gallery selection for display only.
 */
/**
 * Which gallery tab an item belongs to. Mirrors the category split the
 * gallery filters on: `general` is a gallery image, everything else (canvas
 * pixels, control layers, uploads) is an asset.
 */
const getItemGalleryView = (item: GalleryItem): GalleryView => (item.category === 'general' ? 'images' : 'assets');

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
 * Bottom padding the grid surface reserves for the overlaid filmstrip and
 * details island — their collapsed heights plus the 2-unit inset and gap. An
 * expanded Details panel grows over the grid on purpose; it is self-capped at
 * `40cqh` and closes back down.
 */
const getPreviewOverlayReserve = (density: PreviewDensity, hasFilmstrip: boolean): string => {
  if (!hasFilmstrip) {
    return '5.5rem';
  }

  return density === 'full' ? '9.75rem' : '8.75rem';
};

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
  const selectedImageSearch = useMemo(
    () => parseDateTokens(selectedImageQuery.searchTerm),
    [selectedImageQuery.searchTerm]
  );
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
  const navigationBoardId = shouldFollowLive ? activeGalleryPlaceholder.boardId : selectedImageQuery.boardId;
  const navigationGalleryView = shouldFollowLive ? 'images' : selectedImageQuery.galleryView;
  const navigationOrderDir = shouldFollowLive ? imageOrderDir : selectedImageQuery.imageOrderDir;
  const navigationStarredFirst = shouldFollowLive ? starredFirst : selectedImageQuery.starredFirst;
  const hasNavigationContext = shouldFollowLive || hasSelectedItem;
  const { t } = useTranslation();
  const loupeControlsRef = useRef<PreviewLoupeControls | null>(null);
  const videoControllerRef = useRef<PreviewVideoFrameController | null>(null);
  const [copyAvailableItemKey, setCopyAvailableItemKey] = useState<GalleryItemKey | null>(null);
  const navigationContextKey = `${shouldFollowLive}:${selectedItemKey ?? ''}:${navigationBoardId}:${navigationGalleryView}:${navigationOrderDir}:${navigationStarredFirst}:${selectedImageQuery.paginationMode}:${selectedImageQuery.page}:${selectedImageQuery.searchTerm}`;
  const navigationQueryKey = `${shouldFollowLive}:${navigationBoardId}:${navigationGalleryView}:${navigationOrderDir}:${navigationStarredFirst}:${selectedImageQuery.paginationMode}:${selectedImageQuery.searchTerm}`;

  // Lets a boundary fetch that resolves after the user has moved on compare the
  // context it started in against the one now on screen, and drop its stale
  // result. Written from an effect rather than during render: the compiler
  // rejects render-phase ref writes, and an effect event cannot be called from
  // a promise continuation.
  const navigationContextKeyRef = useRef(navigationContextKey);

  useEffect(() => {
    navigationContextKeyRef.current = navigationContextKey;
  }, [navigationContextKey]);

  // A paginated navigation stays anchored to the page the preview opened on,
  // and re-anchors only when the underlying query identity changes. Derived
  // state rather than a ref so the compiler can see the dependency.
  const [navigationAnchor, setNavigationAnchor] = useState({
    page: selectedImageQuery.page,
    queryKey: navigationQueryKey,
  });
  const hasStaleNavigationAnchor = navigationAnchor.queryKey !== navigationQueryKey;

  if (hasStaleNavigationAnchor) {
    setNavigationAnchor({ page: selectedImageQuery.page, queryKey: navigationQueryKey });
  }

  const navigationAnchorPage = hasStaleNavigationAnchor ? selectedImageQuery.page : navigationAnchor.page;

  const boardsQuery = useQuery({
    ...galleryBoardsOptions(),
    enabled: hasSelectedItem,
  });
  const {
    data: boardItemsData,
    fetchNextPage: fetchNextBoardItemsPage,
    fetchPreviousPage: fetchPreviousBoardItemsPage,
    hasNextPage: hasNextBoardItemsPage,
    hasPreviousPage: hasPreviousBoardItemsPage,
    isFetching: isFetchingBoardItems,
    isFetchingNextPage: isFetchingNextBoardItemsPage,
    isFetchingPreviousPage: isFetchingPreviousBoardItemsPage,
  } = useInfiniteQuery({
    ...galleryItemsInfiniteOptions(
      {
        boardId: navigationBoardId,
        createdFrom: shouldFollowLive ? undefined : selectedImageSearch.range?.from,
        createdTo: shouldFollowLive ? undefined : selectedImageSearch.range?.to,
        galleryView: navigationGalleryView,
        orderDir: navigationOrderDir,
        searchTerm: shouldFollowLive ? '' : selectedImageSearch.text,
        starredFirst: navigationStarredFirst,
      },
      !shouldFollowLive && selectedImageQuery.paginationMode === 'paginated'
        ? { kind: 'anchor', offset: navigationAnchorPage * GALLERY_PAGE_SIZE }
        : { kind: 'infinite' }
    ),
    enabled: hasNavigationContext,
  });
  const selectPreviewItem = useCallback(
    (item: GalleryItem) => {
      const itemKey = toGalleryItemKey(item);
      const pageIndex = boardItemsData?.pages.findIndex((page) =>
        page.items.some((candidate) => toGalleryItemKey(candidate) === itemKey)
      );
      const pageParam = pageIndex === undefined || pageIndex < 0 ? undefined : boardItemsData?.pageParams[pageIndex];
      const selectionPage =
        typeof pageParam === 'number' ? Math.floor(pageParam / GALLERY_PAGE_SIZE) : selectedImageQuery.page;

      gallery.selectItem(item, undefined, selectionPage, true);
    },
    [boardItemsData, gallery, selectedImageQuery.page]
  );
  const boards = boardsQuery.data ?? fallbackBoards;
  const optimisticQueueItemIds = useMemo(
    () =>
      new Set(
        queueItems.filter((item) => item.status === 'pending' || item.status === 'running').map((item) => item.id)
      ),
    [queueItems]
  );
  const navigationLocalItems = useMemo(() => {
    const refreshingSelectedSourceId =
      !shouldFollowLive && isFetchingBoardItems && selectedItem?.kind === 'image'
        ? selectedItem.sourceQueueItemId
        : null;

    return localItems.filter(
      (item) =>
        (item.sourceQueueItemId !== undefined && optimisticQueueItemIds.has(item.sourceQueueItemId)) ||
        item.sourceQueueItemId === refreshingSelectedSourceId
    );
  }, [isFetchingBoardItems, localItems, optimisticQueueItemIds, selectedItem, shouldFollowLive]);
  const localBoardItems = useMemo(
    () =>
      getOrderedLocalItems({
        boardId: navigationBoardId,
        galleryView: navigationGalleryView,
        items: navigationLocalItems,
        imageOrderDir: navigationOrderDir,
        starredFirst: navigationStarredFirst,
      }),
    [navigationBoardId, navigationGalleryView, navigationLocalItems, navigationOrderDir, navigationStarredFirst]
  );
  const previewLocalBoardItems = useMemo(() => {
    if (
      shouldFollowLive ||
      !selectedItem ||
      localBoardItems.some((item) => toGalleryItemKey(item) === selectedItemKey)
    ) {
      return localBoardItems;
    }

    return [selectedItem, ...localBoardItems];
  }, [localBoardItems, selectedItem, selectedItemKey, shouldFollowLive]);
  const backendBoardItems = useMemo(() => flattenPreviewItems(boardItemsData), [boardItemsData]);
  const boardItems = useMemo(
    () =>
      !hasNavigationContext
        ? EMPTY_PREVIEW_ITEMS
        : mergePreviewBoardItems(backendBoardItems, previewLocalBoardItems, navigationOrderDir, navigationStarredFirst),
    [backendBoardItems, hasNavigationContext, navigationOrderDir, navigationStarredFirst, previewLocalBoardItems]
  );
  const isLoadingBoard = hasNavigationContext && isFetchingBoardItems;
  const boardName = getBoardName(
    boards,
    displayBoardId,
    t('widgets.gallery.uncategorized'),
    t('widgets.gallery.unknownBoard')
  );
  const navigationSequence = useMemo(
    () =>
      getPreviewNavigationSequence({
        activePlaceholder: activeGalleryPlaceholder,
        boardId: navigationBoardId,
        boardImages: boardItems,
        galleryView: navigationGalleryView,
        imageOrderDir: navigationOrderDir,
        starredFirst: navigationStarredFirst,
      }),
    [
      activeGalleryPlaceholder,
      boardItems,
      navigationBoardId,
      navigationGalleryView,
      navigationOrderDir,
      navigationStarredFirst,
    ]
  );
  const navigationCursor = getPreviewNavigationCursor(navigationSequence, {
    isFollowingLive: shouldFollowLive,
    selectedItemKey,
  });

  // One navigation action shared by the arrow keys and the footer buttons.
  // Compare mode stays inert and never exposes the placeholder.
  const navigate = useCallback(
    (offset: -1 | 1) => {
      if (isComparing) {
        return;
      }

      const target = getPreviewNavigationTarget(navigationSequence, navigationCursor, offset);
      const isAtLoadedBackendBoundary =
        selectedItemKey !== null &&
        (offset === 1
          ? backendBoardItems.at(-1) !== undefined &&
            toGalleryItemKey(backendBoardItems.at(-1)!) === selectedItemKey &&
            hasNextBoardItemsPage
          : backendBoardItems[0] !== undefined &&
            toGalleryItemKey(backendBoardItems[0]) === selectedItemKey &&
            hasPreviousBoardItemsPage);

      if (!isAtLoadedBackendBoundary) {
        if (!target) {
          return;
        }

        if (target.kind === 'item') {
          selectPreviewItem(target.item);
        } else {
          account.updateProjectPreferences({ showProgressImagesInViewer: true });
        }
        return;
      }

      if (offset === 1 ? isFetchingNextBoardItemsPage : isFetchingPreviousBoardItemsPage) {
        return;
      }

      const fetchBoundaryPage = offset === 1 ? fetchNextBoardItemsPage : fetchPreviousBoardItemsPage;

      void fetchBoundaryPage().then((result) => {
        if (result.isError || navigationContextKeyRef.current !== navigationContextKey) {
          return;
        }

        const nextBackendBoardItems = flattenPreviewItems(result.data);
        const nextBoardItems = mergePreviewBoardItems(
          nextBackendBoardItems,
          previewLocalBoardItems,
          navigationOrderDir,
          navigationStarredFirst
        );
        const nextNavigationSequence = getPreviewNavigationSequence({
          activePlaceholder: activeGalleryPlaceholder,
          boardId: navigationBoardId,
          boardImages: nextBoardItems,
          galleryView: navigationGalleryView,
          imageOrderDir: navigationOrderDir,
          starredFirst: navigationStarredFirst,
        });
        const nextNavigationCursor = getPreviewNavigationCursor(nextNavigationSequence, {
          isFollowingLive: shouldFollowLive,
          selectedItemKey,
        });
        const nextTarget = getPreviewNavigationTarget(nextNavigationSequence, nextNavigationCursor, offset);

        if (nextTarget?.kind === 'item') {
          selectPreviewItem(nextTarget.item);
        } else if (nextTarget?.kind === 'placeholder') {
          account.updateProjectPreferences({ showProgressImagesInViewer: true });
        }
      });
    },
    [
      account,
      activeGalleryPlaceholder,
      backendBoardItems,
      fetchNextBoardItemsPage,
      fetchPreviousBoardItemsPage,
      hasNextBoardItemsPage,
      hasPreviousBoardItemsPage,
      isComparing,
      isFetchingNextBoardItemsPage,
      isFetchingPreviousBoardItemsPage,
      navigationBoardId,
      navigationContextKey,
      navigationCursor,
      navigationGalleryView,
      navigationOrderDir,
      navigationSequence,
      navigationStarredFirst,
      previewLocalBoardItems,
      selectedItemKey,
      selectPreviewItem,
      shouldFollowLive,
    ]
  );

  const handleNavigationKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.target instanceof Element && event.target.closest('video')) {
        return;
      }

      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') {
        return;
      }

      if (isComparing) {
        return;
      }

      // stopPropagation keeps the widget hotkey runtime from handling the same
      // arrow press a second time.
      event.preventDefault();
      event.stopPropagation();
      navigate(event.key === 'ArrowLeft' ? -1 : 1);
    },
    [isComparing, navigate]
  );

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

  // Warm the browser cache for the sequence neighbors so arrow-key navigation
  // swaps without a decode flash.
  const previousNeighbor = navigationSequence[navigationCursor - 1];
  const nextNeighbor = navigationSequence[navigationCursor + 1];
  const previousNeighborUrl =
    previousNeighbor?.kind === 'item' && previousNeighbor.item.kind === 'image' ? previousNeighbor.item.fullUrl : null;
  const nextNeighborUrl =
    nextNeighbor?.kind === 'item' && nextNeighbor.item.kind === 'image' ? nextNeighbor.item.fullUrl : null;

  useEffect(() => {
    [previousNeighborUrl, nextNeighborUrl].forEach((url) => {
      if (url) {
        new Image().src = url;
      }
    });
  }, [nextNeighborUrl, previousNeighborUrl]);
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
            placeholder={activeGalleryPlaceholder}
            progressImage={matchingProgressImage}
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
  const { t } = useTranslation();
  const overlayReserve = getPreviewOverlayReserve(density, filmstripItems !== null);

  // The grid surface fills; the filmstrip and details float above its lower
  // edge. The surface reserves matching bottom padding so the fitted media is
  // never tucked under them — only the dot grid passes behind.
  return (
    <Flex direction="column" h="full" minH="0" position="relative" w="full">
      <PreviewFrame
        dragItem={dragItem}
        frameHeight={frameHeight}
        frameWidth={frameWidth}
        isItemCurrent={isItemCurrent}
        isLive={false}
        liveBadgeLabel={t('common.generating')}
        loupeControlsRef={loupeControlsRef}
        onVideoCopyAvailabilityChange={onCopyAvailabilityChange}
        padding={density === 'full' ? '6' : '3'}
        paddingBottom={overlayReserve}
        shouldAntialiasLiveImage
        source={source}
        variant="framed"
        videoControllerRef={videoControllerRef}
        onContextMenu={onContextMenu}
      />
      <Stack bottom="2" gap="2" insetX="2" position="absolute" zIndex="1">
        {filmstripItems ? (
          <PreviewFilmstrip
            density={density}
            items={filmstripItems}
            selectedItemKey={toGalleryItemKey(item)}
            onSelect={onSelectItem}
          />
        ) : null}
        <PreviewFooter
          actionImage={actionImage}
          actions={actions}
          boardItemCount={boardItemCount}
          item={item}
          isLoadingBoard={isLoadingBoard}
          isMetadataOpen={isMetadataOpen}
          selectedIndex={selectedIndex}
          onNext={onNext}
          onPrevious={onPrevious}
          onToggleMetadata={onToggleMetadata}
        />
      </Stack>
    </Flex>
  );
};

const LivePreview = ({
  placeholder,
  progressImage,
  shouldAntialiasProgressImage,
}: {
  placeholder: GalleryQueuePlaceholder;
  progressImage: LatestProgressImageSnapshot | null;
  shouldAntialiasProgressImage: boolean;
}) => {
  const { t } = useTranslation();
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

  return (
    <PreviewFrame
      frameHeight={previewImage?.height ?? placeholder.height}
      frameWidth={previewImage?.width ?? placeholder.width}
      isLive
      liveBadgeLabel={t('common.generating')}
      liveQueueItemId={placeholder.queueItemId}
      shouldAntialiasLiveImage={shouldAntialiasProgressImage}
      source={source}
      variant="inset"
    />
  );
};

/**
 * One tile in the multi-session grid.
 *
 * Subscribes to its own slot's progress image rather than receiving it from the
 * parent, so a frame from one GPU's session re-renders only that tile.
 */
const LivePreviewTile = ({
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

  return (
    <PreviewFrame
      frameHeight={previewImage?.height ?? placeholder.height}
      frameWidth={previewImage?.width ?? placeholder.width}
      isLive
      liveBadgeLabel={
        deviceLabel ? t('widgets.queue.device.shortLabel', { index: deviceLabel.index }) : t('common.generating')
      }
      liveQueueItemId={placeholder.queueItemId}
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
const LivePreviewTiles = ({
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
    <PreviewFrame
      frameHeight={1}
      frameWidth={1}
      isLive={false}
      liveBadgeLabel={t('common.generating')}
      shouldAntialiasLiveImage
      source={null}
      variant="inset"
    >
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

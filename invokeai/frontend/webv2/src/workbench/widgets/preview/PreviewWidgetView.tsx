import type { QueueItem } from '@features/queue/contracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Box, Stack, Text } from '@chakra-ui/react';
import { useDndMonitor, type DragEndEvent } from '@dnd-kit/core';
import {
  galleryImages,
  type GalleryBoard,
  type GalleryImage,
  type GalleryView,
  type GeneratedImageContract,
} from '@features/gallery';
import {
  getGalleryCompareImage,
  getGalleryGenerationSequence,
  getGallerySelectedImageQuery,
  getGallerySettings,
  getBoundedRecentImages,
  normalizeGalleryImage,
  type GalleryQueuePlaceholder,
} from '@features/gallery/contracts';
import {
  flattenGalleryImagesData,
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
  galleryBoardsOptions,
  galleryImagesInfiniteOptions,
} from '@features/gallery/queries';
import { createGenerateFormValuesSelector } from '@features/generation/react';
import { useActiveProgressTarget, useProgressImage, type LatestProgressImageSnapshot } from '@features/queue/react';
import { parseDateTokens } from '@platform/search/dateTokens';
import {
  imageUrlToStreamingSource,
  progressImageToStreamingSource,
} from '@platform/ui/streaming-image/streamingImageSource';
import { useStreamingImageSource } from '@platform/ui/streaming-image/useStreamingImageSource';
import { useInfiniteQuery, useQuery } from '@tanstack/react-query';
import {
  ImageContextMenu,
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
} from '@workbench/WorkbenchContext';
import { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState, type KeyboardEvent, type Ref } from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewLoupeControls } from './usePreviewLoupe';

import { PreviewCompare } from './PreviewCompare';
import { resolvePreviewCompareDrop } from './previewCompareDnd';
import { usePreviewDensity, type PreviewDensity } from './previewDensity';
import { PreviewFilmstrip } from './PreviewFilmstrip';
import { PreviewFooter } from './PreviewFooter';
import { PreviewFrame } from './PreviewFrame';
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

type PreviewImage = GeneratedImageContract & Partial<Pick<GalleryImage, 'boardId' | 'imageCategory' | 'starred'>>;
const EMPTY_PREVIEW_IMAGES: PreviewImage[] = [];

const fallbackBoards: GalleryBoard[] = [
  {
    archived: false,
    assetCount: 0,
    id: 'none',
    imageCount: 0,
    kind: 'uncategorized',
    name: 'Uncategorized',
    videoCount: 0,
  },
];

const getGalleryImages = (values: Record<string, unknown>, queueItems: QueueItem[]): PreviewImage[] => {
  const queueBoardIds = new Map(queueItems.map((item) => [item.id, item.snapshot.galleryBoardId ?? 'none'] as const));

  return getBoundedRecentImages(values.recentImages).map((image) =>
    normalizeGalleryImage(image, queueBoardIds.get(image.sourceQueueItemId))
  );
};

const getSelectedImage = (values: Record<string, unknown>, localImages: PreviewImage[]): PreviewImage | null => {
  const selectedImage = values.selectedImage;
  const selectedImageName = typeof values.selectedImageName === 'string' ? values.selectedImageName : null;

  if (
    selectedImage &&
    typeof selectedImage === 'object' &&
    typeof (selectedImage as GeneratedImageContract).imageName === 'string'
  ) {
    const image = selectedImage as PreviewImage;

    return typeof image.boardId === 'string'
      ? image
      : (localImages.find((candidate) => candidate.imageName === image.imageName) ?? image);
  }

  return localImages.find((image) => image.imageName === selectedImageName) ?? localImages[0] ?? null;
};

const getOrderedPreviewImages = (
  images: PreviewImage[],
  imageOrderDir: 'ASC' | 'DESC',
  starredFirst: boolean,
  inputOrder: 'display' | 'newest-first'
): PreviewImage[] =>
  images
    .map((image, index) => ({ image, index }))
    .sort((a, b) => {
      if (starredFirst && Boolean(a.image.starred) !== Boolean(b.image.starred)) {
        return a.image.starred ? -1 : 1;
      }

      const chronological = a.image.queuedAt.localeCompare(b.image.queuedAt);

      if (chronological !== 0) {
        return imageOrderDir === 'DESC' ? -chronological : chronological;
      }

      return inputOrder === 'newest-first' && imageOrderDir === 'ASC' ? b.index - a.index : a.index - b.index;
    })
    .map(({ image }) => image);

const getOrderedLocalImages = ({
  boardId,
  galleryView,
  images,
  imageOrderDir,
  starredFirst,
}: {
  boardId: string;
  galleryView: GalleryView;
  images: PreviewImage[];
  imageOrderDir: 'ASC' | 'DESC';
  starredFirst: boolean;
}): PreviewImage[] =>
  getOrderedPreviewImages(
    images.filter((image) => getImageBoardId(image) === boardId && getImageGalleryView(image) === galleryView),
    imageOrderDir,
    starredFirst,
    'newest-first'
  );

export const mergePreviewBoardImages = (
  backendImages: PreviewImage[],
  localImages: PreviewImage[],
  imageOrderDir: 'ASC' | 'DESC',
  starredFirst: boolean
): PreviewImage[] => {
  const backendNames = new Set(backendImages.map((image) => image.imageName));
  const missingLocalImages = localImages.filter((image) => !backendNames.has(image.imageName));

  if (missingLocalImages.length === 0) {
    return backendImages.slice(0, GALLERY_MAX_ROWS);
  }

  return getOrderedPreviewImages(
    [...backendImages, ...missingLocalImages],
    imageOrderDir,
    starredFirst,
    'display'
  ).slice(0, GALLERY_MAX_ROWS);
};

/**
 * The board used for the "N of M" index comes from the image itself, never
 * from the gallery's currently selected board — switching boards or changing
 * the board list sort must not reshuffle the preview. Freshly generated local
 * images (no boardId yet) fall back to the gallery selection for display only.
 */
const getImageBoardId = (image: PreviewImage): string => (typeof image.boardId === 'string' ? image.boardId : 'none');

/**
 * Which gallery tab an image belongs to. Mirrors the category split the
 * gallery filters on: `general` is a gallery image, everything else (canvas
 * pixels, control layers, uploads) is an asset. A freshly generated local
 * image carries no category yet and is a gallery image by default.
 */
const getImageGalleryView = (image: PreviewImage): GalleryView =>
  (image.imageCategory ?? 'general') === 'general' ? 'images' : 'assets';

const getDisplayBoardId = (image: PreviewImage, values: Record<string, unknown>): string => {
  if (typeof image.boardId === 'string') {
    return image.boardId;
  }

  return typeof values.selectedBoardId === 'string' ? values.selectedBoardId : 'none';
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
  const { account, gallery, notifications, widgets } = useWorkbenchCommands();
  const { density, rootRef } = usePreviewDensity(region);
  const recentImages = galleryValues.recentImages;
  const localImages = useMemo(() => getGalleryImages({ recentImages }, queueItems), [queueItems, recentImages]);
  const selectedImage = useMemo(() => getSelectedImage(galleryValues, localImages), [galleryValues, localImages]);
  const compareImage = getGalleryCompareImage(galleryValues);
  const comparisonMode = getPreviewComparisonMode(previewValues);
  const displayBoardId = selectedImage ? getDisplayBoardId(selectedImage, galleryValues) : 'none';
  const hasSelectedImage = selectedImage !== null;
  const { imageOrderDir, starredFirst } = getGallerySettings(galleryValues);
  const selectedImageQuery = getGallerySelectedImageQuery(galleryValues);
  const selectedImageSearch = useMemo(
    () => parseDateTokens(selectedImageQuery.searchTerm),
    [selectedImageQuery.searchTerm]
  );
  const selectedImageName = selectedImage?.imageName ?? null;
  const isComparing =
    selectedImage !== null && compareImage !== null && compareImage.imageName !== selectedImage.imageName;
  const activeGalleryPlaceholder = useMemo(
    () => getGalleryGenerationSequence(queueItems, activeProgressTarget).liveSlot,
    [activeProgressTarget, queueItems]
  );
  const matchingProgressImage = getMatchingProgressImage(progressImage, activeGalleryPlaceholder);
  const shouldFollowLive = showProgressImagesInViewer && activeGalleryPlaceholder !== null && !isComparing;
  const navigationBoardId = shouldFollowLive ? activeGalleryPlaceholder.boardId : selectedImageQuery.boardId;
  const navigationGalleryView = shouldFollowLive ? 'images' : selectedImageQuery.galleryView;
  const navigationOrderDir = shouldFollowLive ? imageOrderDir : selectedImageQuery.imageOrderDir;
  const navigationStarredFirst = shouldFollowLive ? starredFirst : selectedImageQuery.starredFirst;
  const hasNavigationContext = shouldFollowLive || hasSelectedImage;
  const { t } = useTranslation();
  const loupeControlsRef = useRef<PreviewLoupeControls | null>(null);
  const navigationContextKey = `${shouldFollowLive}:${selectedImageName ?? ''}:${navigationBoardId}:${navigationGalleryView}:${navigationOrderDir}:${navigationStarredFirst}:${selectedImageQuery.paginationMode}:${selectedImageQuery.page}:${selectedImageQuery.searchTerm}`;
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
    enabled: hasSelectedImage,
  });
  const {
    data: boardImagesData,
    fetchNextPage: fetchNextBoardImagesPage,
    fetchPreviousPage: fetchPreviousBoardImagesPage,
    hasNextPage: hasNextBoardImagesPage,
    hasPreviousPage: hasPreviousBoardImagesPage,
    isFetching: isFetchingBoardImages,
    isFetchingNextPage: isFetchingNextBoardImagesPage,
    isFetchingPreviousPage: isFetchingPreviousBoardImagesPage,
  } = useInfiniteQuery({
    ...galleryImagesInfiniteOptions(
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
  const selectPreviewImage = useCallback(
    (image: GeneratedImageContract) => {
      const pageIndex = boardImagesData?.pages.findIndex((page) =>
        page.images.some((candidate) => candidate.imageName === image.imageName)
      );
      const pageParam = pageIndex === undefined || pageIndex < 0 ? undefined : boardImagesData?.pageParams[pageIndex];
      const selectionPage =
        typeof pageParam === 'number' ? Math.floor(pageParam / GALLERY_PAGE_SIZE) : selectedImageQuery.page;

      gallery.selectImage(image, undefined, selectionPage, true);
    },
    [boardImagesData, gallery, selectedImageQuery.page]
  );
  const boards = boardsQuery.data ?? fallbackBoards;
  const optimisticQueueItemIds = useMemo(
    () =>
      new Set(
        queueItems.filter((item) => item.status === 'pending' || item.status === 'running').map((item) => item.id)
      ),
    [queueItems]
  );
  const navigationLocalImages = useMemo(() => {
    const refreshingSelectedSourceId =
      !shouldFollowLive && isFetchingBoardImages ? selectedImage?.sourceQueueItemId : null;

    return localImages.filter(
      (image) =>
        optimisticQueueItemIds.has(image.sourceQueueItemId) || image.sourceQueueItemId === refreshingSelectedSourceId
    );
  }, [isFetchingBoardImages, localImages, optimisticQueueItemIds, selectedImage?.sourceQueueItemId, shouldFollowLive]);
  const localBoardImages = useMemo(
    () =>
      getOrderedLocalImages({
        boardId: navigationBoardId,
        galleryView: navigationGalleryView,
        images: navigationLocalImages,
        imageOrderDir: navigationOrderDir,
        starredFirst: navigationStarredFirst,
      }),
    [navigationBoardId, navigationGalleryView, navigationLocalImages, navigationOrderDir, navigationStarredFirst]
  );
  const previewLocalBoardImages = useMemo(() => {
    if (
      shouldFollowLive ||
      !selectedImage ||
      localBoardImages.some((image) => image.imageName === selectedImage.imageName)
    ) {
      return localBoardImages;
    }

    return [selectedImage, ...localBoardImages];
  }, [localBoardImages, selectedImage, shouldFollowLive]);
  const backendBoardImages = useMemo(() => flattenGalleryImagesData(boardImagesData), [boardImagesData]);
  const boardImages = useMemo(
    () =>
      !hasNavigationContext
        ? EMPTY_PREVIEW_IMAGES
        : mergePreviewBoardImages(
            backendBoardImages,
            previewLocalBoardImages,
            navigationOrderDir,
            navigationStarredFirst
          ),
    [backendBoardImages, hasNavigationContext, navigationOrderDir, navigationStarredFirst, previewLocalBoardImages]
  );
  const isLoadingBoard = hasNavigationContext && isFetchingBoardImages;
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
        boardImages,
        galleryView: navigationGalleryView,
        imageOrderDir: navigationOrderDir,
        starredFirst: navigationStarredFirst,
      }),
    [
      activeGalleryPlaceholder,
      boardImages,
      navigationBoardId,
      navigationGalleryView,
      navigationOrderDir,
      navigationStarredFirst,
    ]
  );
  const navigationCursor = getPreviewNavigationCursor(navigationSequence, {
    isFollowingLive: shouldFollowLive,
    selectedImageName,
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
        selectedImageName !== null &&
        (offset === 1
          ? backendBoardImages.at(-1)?.imageName === selectedImageName && hasNextBoardImagesPage
          : backendBoardImages[0]?.imageName === selectedImageName && hasPreviousBoardImagesPage);

      if (!isAtLoadedBackendBoundary) {
        if (!target) {
          return;
        }

        if (target.kind === 'image') {
          selectPreviewImage(target.image);
        } else {
          account.updateProjectPreferences({ showProgressImagesInViewer: true });
        }
        return;
      }

      if (offset === 1 ? isFetchingNextBoardImagesPage : isFetchingPreviousBoardImagesPage) {
        return;
      }

      const fetchBoundaryPage = offset === 1 ? fetchNextBoardImagesPage : fetchPreviousBoardImagesPage;

      void fetchBoundaryPage().then((result) => {
        if (result.isError || navigationContextKeyRef.current !== navigationContextKey) {
          return;
        }

        const nextBackendBoardImages = flattenGalleryImagesData(result.data);
        const nextBoardImages = mergePreviewBoardImages(
          nextBackendBoardImages,
          previewLocalBoardImages,
          navigationOrderDir,
          navigationStarredFirst
        );
        const nextNavigationSequence = getPreviewNavigationSequence({
          activePlaceholder: activeGalleryPlaceholder,
          boardId: navigationBoardId,
          boardImages: nextBoardImages,
          galleryView: navigationGalleryView,
          imageOrderDir: navigationOrderDir,
          starredFirst: navigationStarredFirst,
        });
        const nextNavigationCursor = getPreviewNavigationCursor(nextNavigationSequence, {
          isFollowingLive: shouldFollowLive,
          selectedImageName,
        });
        const nextTarget = getPreviewNavigationTarget(nextNavigationSequence, nextNavigationCursor, offset);

        if (nextTarget?.kind === 'image') {
          selectPreviewImage(nextTarget.image);
        } else if (nextTarget?.kind === 'placeholder') {
          account.updateProjectPreferences({ showProgressImagesInViewer: true });
        }
      });
    },
    [
      account,
      activeGalleryPlaceholder,
      backendBoardImages,
      fetchNextBoardImagesPage,
      fetchPreviousBoardImagesPage,
      hasNextBoardImagesPage,
      hasPreviousBoardImagesPage,
      isComparing,
      isFetchingNextBoardImagesPage,
      isFetchingPreviousBoardImagesPage,
      navigationBoardId,
      navigationContextKey,
      navigationCursor,
      navigationGalleryView,
      navigationOrderDir,
      navigationSequence,
      navigationStarredFirst,
      previewLocalBoardImages,
      selectedImageName,
      selectPreviewImage,
      shouldFollowLive,
    ]
  );

  const handleNavigationKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
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
  const onImagesDeleted = useCallback(
    (imageNames: string[]) => {
      const deletedNames = new Set(imageNames);
      const images = boardImages;
      const anchorName = selectedImage?.imageName ?? null;

      if (!anchorName || !deletedNames.has(anchorName)) {
        return;
      }

      const anchorIndex = images.findIndex((image) => image.imageName === anchorName);

      if (anchorIndex === -1) {
        return;
      }

      const remaining = images.filter((image) => !deletedNames.has(image.imageName));
      const remainingBeforeAnchor = images
        .slice(0, anchorIndex)
        .filter((image) => !deletedNames.has(image.imageName)).length;
      const nextImage = remaining[remainingBeforeAnchor] ?? remaining[remainingBeforeAnchor - 1] ?? null;

      if (nextImage) {
        selectPreviewImage(nextImage);
      }
    },
    [boardImages, selectPreviewImage, selectedImage]
  );
  const projectId = useActiveProjectId();
  const imageActions = useImageActions({
    boards,
    generateValues,
    onImagesDeleted,
    projectId,
  });
  const contextMenuImage = useMemo<GalleryImage | null>(() => {
    if (!selectedImage) {
      return null;
    }

    const boardImage = boardImages.find((image) => image.imageName === selectedImage.imageName);

    return {
      ...selectedImage,
      boardId: boardImage?.boardId ?? displayBoardId,
      imageCategory: boardImage?.imageCategory ?? selectedImage.imageCategory ?? 'general',
      starred: boardImage?.starred ?? selectedImage.starred ?? false,
    };
  }, [boardImages, displayBoardId, selectedImage]);
  const exitCompare = useCallback(() => gallery.setCompareImage(null), [gallery]);
  const swapCompareImages = useCallback(() => {
    if (selectedImage && compareImage) {
      selectPreviewImage(compareImage);
      gallery.setCompareImage(selectedImage);
    }
  }, [compareImage, gallery, selectPreviewImage, selectedImage]);
  const setComparisonMode = useCallback(
    (comparisonMode: PreviewComparisonMode) => widgets.patchValues('preview', { comparisonMode }),
    [widgets]
  );
  const isMetadataOpen = getPreviewMetadataOpen(previewValues);
  const toggleMetadata = useCallback(
    () => widgets.patchValues('preview', { metadataOpen: !isMetadataOpen }),
    [isMetadataOpen, widgets]
  );
  const isFilmstripVisible = getPreviewFilmstripVisible(previewValues);
  const selectImage = selectPreviewImage;

  // Drop-to-compare: any gallery-image drag dropped on the frame's drop zone
  // arms that image for comparison. The drag payload only carries names, so
  // the full contract is fetched before dispatching.
  const handleCompareDrop = useCallback(
    (event: DragEndEvent) => {
      const resolution = resolvePreviewCompareDrop(event.active.data.current, event.over?.data.current ?? null);

      if (!resolution) {
        return;
      }

      // Prefer images we already hold (board context includes fresh local
      // generations that would 404 on a backend by-name fetch).
      const localImage =
        boardImages.find((image) => image.imageName === resolution.imageName) ??
        (selectedImage?.imageName === resolution.imageName ? selectedImage : null);

      if (localImage) {
        gallery.setCompareImage(localImage);
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
    [boardImages, gallery, notifications, selectedImage]
  );
  useDndMonitor({ onDragEnd: handleCompareDrop });
  const openImageContextMenu = useCallback(
    (x: number, y: number) => {
      if (contextMenuImage) {
        setContextMenuTarget({ images: [contextMenuImage], x, y });
      }
    },
    [contextMenuImage]
  );
  const selectNextImage = useCallback(() => navigate(1), [navigate]);
  const selectPreviousImage = useCallback(() => navigate(-1), [navigate]);
  const closeContextMenu = useCallback(() => setContextMenuTarget(null), []);
  const headerImageName = shouldFollowLive ? null : selectedImageName;

  // Publish the header chrome context (the "[board] / [image]" label and the
  // action strip's image + actions) for the widget frame; the chrome renders
  // outside this view, so an external store is the sync channel. Cleared on
  // unmount so stale chrome never outlives us.
  useEffect(() => {
    previewHeaderStore.set({
      actionImage: shouldFollowLive ? null : contextMenuImage,
      actions: shouldFollowLive ? null : imageActions,
      boardName: headerImageName === null ? null : boardName,
      imageName: headerImageName,
      openImageMenu: shouldFollowLive ? null : openImageContextMenu,
    });
  }, [boardName, contextMenuImage, headerImageName, imageActions, openImageContextMenu, shouldFollowLive]);

  useEffect(() => () => previewHeaderStore.clear(), []);

  // Warm the browser cache for the sequence neighbors so arrow-key navigation
  // swaps without a decode flash.
  const previousNeighbor = navigationSequence[navigationCursor - 1];
  const nextNeighbor = navigationSequence[navigationCursor + 1];
  const previousNeighborUrl = previousNeighbor?.kind === 'image' ? previousNeighbor.image.imageUrl : null;
  const nextNeighborUrl = nextNeighbor?.kind === 'image' ? nextNeighbor.image.imageUrl : null;

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

    if (commandId === 'viewer.swapImages' && selectedImage && compareImage) {
      selectPreviewImage(compareImage);
      gallery.setCompareImage(selectedImage);
      return;
    }

    if (commandId === 'viewer.deleteImage' && selectedImage && !shouldFollowLive) {
      void imageActions.deleteImages([selectedImage.imageName]);
      return;
    }

    if (commandId === 'viewer.zoomToActual') {
      loupeControlsRef.current?.zoomToActual();
      return;
    }

    if (commandId === 'viewer.zoomToFit') {
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
      ['viewer.swapImages', t('widgets.preview.commands.swapComparisonImages'), ['c']],
      ['viewer.deleteImage', t('widgets.preview.commands.deletePreviewImage'), ['delete', 'backspace']],
      ['viewer.zoomToActual', t('widgets.preview.commands.zoomToActual'), ['1']],
      ['viewer.zoomToFit', t('widgets.preview.commands.zoomToFit'), ['f']],
      ['viewer.toggleFilmstrip', t('widgets.preview.commands.toggleFilmstrip'), ['t']],
    ] as const;
    const disposers = hotkeys.flatMap(([id, title, defaultKeys]) => [
      runtime.commands.register({ handler: () => executeViewerHotkey(id), id, title }),
      runtime.hotkeys.register({ commandId: id, defaultKeys: [...defaultKeys], id, title }),
    ]);

    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [runtime.commands, runtime.hotkeys, t]);

  return (
    <Box ref={rootRef} p="2" h="full">
      {/* Single always-mounted keyboard boundary: DOM focus survives swaps
          between the live, selected, and compare branches, so arrow
          navigation keeps working across them. */}
      <Stack
        aria-label={t('widgets.labels.preview')}
        gap="3"
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
            placeholder={activeGalleryPlaceholder}
            progressImage={matchingProgressImage}
            shouldAntialiasProgressImage={antialiasProgressImages}
          />
        ) : selectedImage ? (
          <>
            {isComparing && compareImage ? (
              <PreviewCompare
                baseImage={selectedImage}
                compareImage={compareImage}
                mode={comparisonMode}
                runtime={runtime}
                onExit={exitCompare}
                onModeChange={setComparisonMode}
                onSwap={swapCompareImages}
              />
            ) : (
              <SelectedImagePreview
                actionImage={contextMenuImage}
                actions={imageActions}
                filmstripImages={isFilmstripVisible && density !== 'minimal' ? boardImages : null}
                loupeControlsRef={loupeControlsRef}
                boardImageCount={navigationSequence.length}
                density={density}
                image={selectedImage}
                isLoadingBoard={isLoadingBoard}
                isMetadataOpen={isMetadataOpen}
                selectedIndex={navigationCursor}
                onContextMenu={openImageContextMenu}
                onNext={selectNextImage}
                onPrevious={selectPreviousImage}
                onSelectImage={selectImage}
                onToggleMetadata={toggleMetadata}
              />
            )}
            <ImageContextMenu
              actions={imageActions}
              boards={boards}
              target={contextMenuTarget}
              onClose={closeContextMenu}
            />
          </>
        ) : (
          <EmptyPreview />
        )}
      </Stack>
    </Box>
  );
};

const SelectedImagePreview = ({
  actionImage,
  actions,
  boardImageCount,
  density,
  filmstripImages,
  image,
  isLoadingBoard,
  isMetadataOpen,
  loupeControlsRef,
  selectedIndex,
  onContextMenu,
  onNext,
  onPrevious,
  onSelectImage,
  onToggleMetadata,
}: {
  actionImage: GalleryImage | null;
  actions: ImageActions;
  boardImageCount: number;
  density: PreviewDensity;
  /** Board thumbnails for the filmstrip, or null when the strip is hidden. */
  filmstripImages: PreviewImage[] | null;
  image: GeneratedImageContract;
  isLoadingBoard: boolean;
  isMetadataOpen: boolean;
  loupeControlsRef?: Ref<PreviewLoupeControls>;
  selectedIndex: number;
  onContextMenu: (x: number, y: number) => void;
  onNext: () => void;
  onPrevious: () => void;
  onSelectImage: (image: GeneratedImageContract) => void;
  onToggleMetadata: () => void;
}) => {
  const { t } = useTranslation();
  const previewImage = useStreamingImageSource({
    fallbackImage: imageUrlToStreamingSource({
      alt: image.imageName,
      height: image.height,
      kind: 'fallback',
      src: image.imageUrl,
      width: image.width,
    }),
  });
  const previewWidth = previewImage?.width ?? image.width;
  const previewHeight = previewImage?.height ?? image.height;
  const dragImage = useMemo(
    () => ({ boardId: actionImage?.boardId ?? 'none', imageName: image.imageName }),
    [actionImage?.boardId, image.imageName]
  );
  return (
    <Stack gap="2" h="full" minH="0" w="full">
      <PreviewFrame
        dragImage={dragImage}
        frameHeight={previewHeight}
        frameWidth={previewWidth}
        isLive={false}
        liveBadgeLabel={t('common.generating')}
        loupeControlsRef={loupeControlsRef}
        padding={density === 'full' ? '6' : '3'}
        shouldAntialiasLiveImage
        source={previewImage}
        variant="framed"
        onContextMenu={onContextMenu}
      />
      {filmstripImages ? (
        <PreviewFilmstrip
          density={density}
          images={filmstripImages}
          selectedImageName={image.imageName}
          onSelect={onSelectImage}
        />
      ) : null}
      <PreviewFooter
        actionImage={actionImage}
        actions={actions}
        boardImageCount={boardImageCount}
        image={image}
        isLive={false}
        isLoadingBoard={isLoadingBoard}
        isMetadataOpen={isMetadataOpen}
        selectedIndex={selectedIndex}
        onNext={onNext}
        onPrevious={onPrevious}
        onToggleMetadata={onToggleMetadata}
      />
    </Stack>
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

  return (
    <PreviewFrame
      frameHeight={previewImage?.height ?? placeholder.height}
      frameWidth={previewImage?.width ?? placeholder.width}
      isLive
      liveBadgeLabel={t('common.generating')}
      liveQueueItemId={placeholder.queueItemId}
      shouldAntialiasLiveImage={shouldAntialiasProgressImage}
      source={previewImage}
      variant="inset"
    />
  );
};

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

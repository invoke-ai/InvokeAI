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
  getGalleryImagesRefreshToken,
  getGalleryRecentImagesKey,
  getGalleryRefreshToken,
  getGallerySettings,
  normalizeGalleryImage,
  type GalleryQueuePlaceholder,
} from '@features/gallery/contracts';
import { galleryBoardsOptions, galleryImagesOptions } from '@features/gallery/queries';
import { createGenerateFormValuesSelector } from '@features/generation/react';
import { useActiveProgressTarget, useProgressImage, type LatestProgressImageSnapshot } from '@features/queue/react';
import {
  imageUrlToStreamingSource,
  progressImageToStreamingSource,
} from '@platform/ui/streaming-image/streamingImageSource';
import { useStreamingImageSource } from '@platform/ui/streaming-image/useStreamingImageSource';
import { useQuery } from '@tanstack/react-query';
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
  { archived: false, assetCount: 0, id: 'none', imageCount: 0, kind: 'uncategorized', name: 'Uncategorized' },
];

const getGalleryImages = (values: Record<string, unknown>, queueItems: QueueItem[]): PreviewImage[] =>
  Array.isArray(values.recentImages)
    ? (() => {
        const queueBoardIds = new Map(
          queueItems.map((item) => [item.id, item.snapshot.galleryBoardId ?? 'none'] as const)
        );

        return (values.recentImages as Array<GeneratedImageContract & Partial<GalleryImage>>).map((image) =>
          normalizeGalleryImage(image, queueBoardIds.get(image.sourceQueueItemId))
        );
      })()
    : [];

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
    return backendImages;
  }

  return getOrderedPreviewImages([...backendImages, ...missingLocalImages], imageOrderDir, starredFirst, 'display');
};

/**
 * The board used for the "N of M" index comes from the image itself, never
 * from the gallery's currently selected board — switching boards or changing
 * the board list sort must not reshuffle the preview. Freshly generated local
 * images (no boardId yet) fall back to the gallery selection for display only.
 */
const getImageBoardId = (image: PreviewImage): string => (typeof image.boardId === 'string' ? image.boardId : 'none');

const getDisplayBoardId = (image: PreviewImage, values: Record<string, unknown>): string => {
  if (typeof image.boardId === 'string') {
    return image.boardId;
  }

  return typeof values.selectedBoardId === 'string' ? values.selectedBoardId : 'none';
};

const getImageGalleryView = (image: PreviewImage): GalleryView => {
  if (
    image.imageCategory === 'control' ||
    image.imageCategory === 'mask' ||
    image.imageCategory === 'user' ||
    image.imageCategory === 'other'
  ) {
    return 'assets';
  }

  return 'images';
};

const getBoardName = (
  boards: GalleryBoard[],
  boardId: string,
  uncategorizedLabel: string,
  unknownBoardLabel: string
): string =>
  boardId === 'none' ? uncategorizedLabel : (boards.find((board) => board.id === boardId)?.name ?? unknownBoardLabel);

export const getPreviewBoardsRequestKey = ({
  hasSelectedImage,
  refreshToken,
}: {
  hasSelectedImage: boolean;
  refreshToken: string;
}): string | null => (hasSelectedImage ? refreshToken : null);

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
  const recentImages = Array.isArray(galleryValues.recentImages) ? galleryValues.recentImages : null;
  const localImages = useMemo(() => getGalleryImages({ recentImages }, queueItems), [queueItems, recentImages]);
  const selectedImage = useMemo(() => getSelectedImage(galleryValues, localImages), [galleryValues, localImages]);
  const compareImage = getGalleryCompareImage(galleryValues);
  const comparisonMode = getPreviewComparisonMode(previewValues);
  const imageBoardId = selectedImage ? getImageBoardId(selectedImage) : 'none';
  const displayBoardId = selectedImage ? getDisplayBoardId(selectedImage, galleryValues) : 'none';
  const selectedGalleryView = selectedImage ? getImageGalleryView(selectedImage) : 'images';
  const hasSelectedImage = selectedImage !== null;
  const { imageOrderDir, starredFirst } = getGallerySettings(galleryValues);
  const selectedImageName = selectedImage?.imageName ?? null;
  const isComparing =
    selectedImage !== null && compareImage !== null && compareImage.imageName !== selectedImage.imageName;
  const activeGalleryPlaceholder = useMemo(
    () => getGalleryGenerationSequence(queueItems, activeProgressTarget).liveSlot,
    [activeProgressTarget, queueItems]
  );
  const matchingProgressImage = getMatchingProgressImage(progressImage, activeGalleryPlaceholder);
  const shouldFollowLive = showProgressImagesInViewer && activeGalleryPlaceholder !== null && !isComparing;
  const navigationBoardId = shouldFollowLive ? activeGalleryPlaceholder.boardId : imageBoardId;
  const navigationGalleryView = shouldFollowLive ? 'images' : selectedGalleryView;
  const hasNavigationContext = shouldFollowLive || hasSelectedImage;
  const boardRefreshToken = getGalleryRefreshToken(galleryValues);
  const imageRefreshToken = getGalleryImagesRefreshToken(galleryValues);
  const recentImagesKey = getGalleryRecentImagesKey(galleryValues);
  const { t } = useTranslation();
  const loupeControlsRef = useRef<PreviewLoupeControls | null>(null);

  const boardsRequestKey = getPreviewBoardsRequestKey({
    hasSelectedImage,
    refreshToken: boardRefreshToken,
  });
  const boardImagesRequestKey = `${navigationBoardId}:${navigationGalleryView}:${imageOrderDir}:${starredFirst}:${imageRefreshToken}:${recentImagesKey}`;
  const boardsQuery = useQuery({
    ...galleryBoardsOptions({ revision: boardsRequestKey ?? undefined }),
    enabled: boardsRequestKey !== null,
  });
  const boardImagesQuery = useQuery({
    ...galleryImagesOptions({
      boardId: navigationBoardId,
      galleryView: navigationGalleryView,
      orderDir: imageOrderDir,
      revision: boardImagesRequestKey,
      searchTerm: '',
      starredFirst,
    }),
    enabled: hasNavigationContext,
  });
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
      !shouldFollowLive && boardImagesQuery.isFetching ? selectedImage?.sourceQueueItemId : null;

    return localImages.filter(
      (image) =>
        optimisticQueueItemIds.has(image.sourceQueueItemId) || image.sourceQueueItemId === refreshingSelectedSourceId
    );
  }, [
    boardImagesQuery.isFetching,
    localImages,
    optimisticQueueItemIds,
    selectedImage?.sourceQueueItemId,
    shouldFollowLive,
  ]);
  const localBoardImages = useMemo(
    () =>
      getOrderedLocalImages({
        boardId: navigationBoardId,
        galleryView: navigationGalleryView,
        images: navigationLocalImages,
        imageOrderDir,
        starredFirst,
      }),
    [imageOrderDir, navigationBoardId, navigationGalleryView, navigationLocalImages, starredFirst]
  );
  const backendBoardImages = boardImagesQuery.data?.images ?? EMPTY_PREVIEW_IMAGES;
  const boardImages = useMemo(
    () =>
      !hasNavigationContext
        ? EMPTY_PREVIEW_IMAGES
        : mergePreviewBoardImages(backendBoardImages, localBoardImages, imageOrderDir, starredFirst),
    [backendBoardImages, hasNavigationContext, imageOrderDir, localBoardImages, starredFirst]
  );
  const isLoadingBoard = hasNavigationContext && boardImagesQuery.isFetching;
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
        imageOrderDir,
        starredFirst,
      }),
    [activeGalleryPlaceholder, boardImages, imageOrderDir, navigationBoardId, navigationGalleryView, starredFirst]
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

      if (!target) {
        return;
      }

      if (target.kind === 'image') {
        gallery.selectImage(target.image);
      } else {
        account.updateProjectPreferences({ showProgressImagesInViewer: true });
      }
    },
    [account, gallery, isComparing, navigationCursor, navigationSequence]
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
        gallery.selectImage(nextImage);
      }
    },
    [boardImages, gallery, selectedImage]
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
      gallery.selectImage(compareImage);
      gallery.setCompareImage(selectedImage);
    }
  }, [compareImage, gallery, selectedImage]);
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
  const selectImage = useCallback((image: GeneratedImageContract) => gallery.selectImage(image), [gallery]);

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
      gallery.selectImage(compareImage);
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
    <Stack gap="3" h="full" minH="0" w="full">
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
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.preview.emptyDescription')}
        </Text>
      </Stack>
    </PreviewFrame>
  );
};

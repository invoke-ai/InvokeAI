/* eslint-disable react/react-compiler */
import type { GeneratedImageContract, WidgetViewProps } from '@workbench/types';

import { Box, Stack, Text } from '@chakra-ui/react';
import { useDndMonitor, type DragEndEvent } from '@dnd-kit/core';
import { useProgressImage, type LatestProgressImageSnapshot } from '@workbench/backend/progressImageStore';
import { useRevealHold } from '@workbench/backend/revealHoldStore';
import {
  getGalleryImageByName,
  listGalleryBoards,
  listGalleryImages,
  type GalleryBoard,
  type GalleryImage,
  type GalleryView,
} from '@workbench/gallery/api';
import { getGallerySettings } from '@workbench/gallery/settings';
import {
  ImageContextMenu,
  useImageActions,
  type ImageActions,
  type ImageContextMenuTarget,
} from '@workbench/image-actions';
import { imageUrlToStreamingSource, progressImageToStreamingSource } from '@workbench/images/streamingImageSource';
import { useStreamingImageSource } from '@workbench/images/useStreamingImageSource';
import {
  getGalleryCompareImage,
  getGalleryImagesRefreshToken,
  getGalleryRecentImagesKey,
  getGalleryRefreshToken,
} from '@workbench/widgets/gallery/galleryStateView';
import { createGenerateFormValuesSelector } from '@workbench/widgets/generate/generateFormViewModel';
import { getProjectWidgetValues } from '@workbench/widgetState';
import {
  useActiveProjectId,
  useActiveProjectSelector,
  useWidgetValuesSelector,
  useWorkbenchDispatch,
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
  getPreviewComparisonMode,
  getPreviewFilmstripVisible,
  getPreviewMetadataOpen,
  type PreviewComparisonMode,
} from './previewSettings';

type PreviewImage = GeneratedImageContract & Partial<Pick<GalleryImage, 'boardId' | 'imageCategory' | 'starred'>>;

const fallbackBoards: GalleryBoard[] = [
  { archived: false, assetCount: 0, id: 'none', imageCount: 0, kind: 'uncategorized', name: 'Uncategorized' },
];

const getGalleryImages = (values: Record<string, unknown>): PreviewImage[] =>
  Array.isArray(values.recentImages)
    ? (values.recentImages as GeneratedImageContract[]).map((image) => ({
        ...image,
        boardId: 'none',
        imageCategory: 'general' as const,
      }))
    : [];

const getSelectedImage = (values: Record<string, unknown>): PreviewImage | null => {
  const images = getGalleryImages(values);
  const selectedImage = values.selectedImage;
  const selectedImageName = typeof values.selectedImageName === 'string' ? values.selectedImageName : null;

  if (
    selectedImage &&
    typeof selectedImage === 'object' &&
    typeof (selectedImage as GeneratedImageContract).imageName === 'string'
  ) {
    return selectedImage as PreviewImage;
  }

  return images.find((image) => image.imageName === selectedImageName) ?? images[0] ?? null;
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

const shouldLoadBackendBoard = (image: PreviewImage): boolean => image.sourceQueueItemId === 'backend-gallery';

export const getPreviewBoardsRequestKey = ({
  hasSelectedImage,
  refreshToken,
}: {
  hasSelectedImage: boolean;
  isBackendImage: boolean;
  refreshToken: string;
}): string | null => (hasSelectedImage ? refreshToken : null);

const selectGenerateRecallValues = createGenerateFormValuesSelector();

export const PreviewWidgetView = ({ region, runtime }: WidgetViewProps) => {
  const galleryValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'gallery'));
  const previewValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'preview'));
  const generateValues = useWidgetValuesSelector('generate', selectGenerateRecallValues);
  const { antialiasProgressImages, showProgressImagesInViewer } = useActiveProjectSelector(
    (project) => project.settings
  );
  const progressImage = useProgressImage();
  const dispatch = useWorkbenchDispatch();
  const { density, rootRef } = usePreviewDensity(region);
  const selectedImage = useMemo(() => getSelectedImage(galleryValues), [galleryValues]);
  const compareImage = getGalleryCompareImage(galleryValues);
  const comparisonMode = getPreviewComparisonMode(previewValues);
  const recentImages = Array.isArray(galleryValues.recentImages) ? galleryValues.recentImages : null;
  const localImages = useMemo(() => getGalleryImages({ recentImages }), [recentImages]);
  const imageBoardId = selectedImage ? getImageBoardId(selectedImage) : 'none';
  const displayBoardId = selectedImage ? getDisplayBoardId(selectedImage, galleryValues) : 'none';
  const selectedGalleryView = selectedImage ? getImageGalleryView(selectedImage) : 'images';
  const hasSelectedImage = selectedImage !== null;
  const isBackendImage = selectedImage ? shouldLoadBackendBoard(selectedImage) : false;
  const { imageOrderDir, starredFirst } = getGallerySettings(galleryValues);
  const boardRefreshToken = getGalleryRefreshToken(galleryValues);
  const imageRefreshToken = getGalleryImagesRefreshToken(galleryValues);
  const recentImagesKey = getGalleryRecentImagesKey(galleryValues);
  const [boards, setBoards] = useState<GalleryBoard[]>(fallbackBoards);
  const [boardImages, setBoardImages] = useState<PreviewImage[]>(localImages);
  const [isLoadingBoard, setIsLoadingBoard] = useState(false);
  const { t } = useTranslation();
  const selectedIndex = selectedImage
    ? boardImages.findIndex((image) => image.imageName === selectedImage.imageName)
    : -1;
  const boardName = getBoardName(
    boards,
    displayBoardId,
    t('widgets.gallery.uncategorized'),
    t('widgets.gallery.unknownBoard')
  );
  const selectedImageRef = useRef(selectedImage);
  const boardImagesRef = useRef(boardImages);
  const loupeControlsRef = useRef<PreviewLoupeControls | null>(null);

  selectedImageRef.current = selectedImage;
  boardImagesRef.current = boardImages;

  const boardsRequestKey = getPreviewBoardsRequestKey({
    hasSelectedImage,
    isBackendImage,
    refreshToken: boardRefreshToken,
  });

  useEffect(() => {
    if (boardsRequestKey === null) {
      setBoards(fallbackBoards);
      return;
    }

    let isStale = false;

    listGalleryBoards()
      .then((nextBoards) => {
        if (!isStale) {
          setBoards(nextBoards);
        }
      })
      .catch(() => {
        if (!isStale) {
          setBoards(fallbackBoards);
        }
      });

    return () => {
      isStale = true;
    };
  }, [boardsRequestKey]);

  // Deliberately NOT keyed on the selected image object or the gallery's
  // board selection/sort: the board context only refetches when the image's
  // own board, the view, the image ordering, or backend contents change.
  useEffect(() => {
    if (!hasSelectedImage) {
      setBoardImages([]);
      setIsLoadingBoard(false);
      return;
    }

    if (!isBackendImage) {
      setBoardImages(localImages);
      setIsLoadingBoard(false);
      return;
    }

    let isStale = false;

    setIsLoadingBoard(true);
    listGalleryImages({
      boardId: imageBoardId,
      galleryView: selectedGalleryView,
      orderDir: imageOrderDir,
      searchTerm: '',
      starredFirst,
    })
      .then((nextImagesPage) => {
        if (isStale) {
          return;
        }

        const fallbackImage = selectedImageRef.current;

        setBoardImages(nextImagesPage.images.length ? nextImagesPage.images : fallbackImage ? [fallbackImage] : []);
      })
      .catch((error: unknown) => {
        if (!isStale) {
          dispatch({
            area: 'preview-board-images',
            message: error instanceof Error ? error.message : String(error),
            namespace: 'gallery',
            type: 'recordError',
          });

          const fallbackImage = selectedImageRef.current;

          setBoardImages(fallbackImage ? [fallbackImage] : []);
        }
      })
      .finally(() => {
        if (!isStale) {
          setIsLoadingBoard(false);
        }
      });

    return () => {
      isStale = true;
    };
  }, [
    dispatch,
    hasSelectedImage,
    imageBoardId,
    imageOrderDir,
    imageRefreshToken,
    isBackendImage,
    localImages,
    recentImagesKey,
    selectedGalleryView,
    starredFirst,
  ]);

  const selectByOffset = useCallback(
    (offset: -1 | 1) => {
      if (selectedIndex === -1) {
        return;
      }

      const nextImage = boardImages[selectedIndex + offset];

      if (nextImage) {
        dispatch({ image: nextImage, type: 'selectGalleryImage' });
      }
    },
    [boardImages, dispatch, selectedIndex]
  );

  const [contextMenuTarget, setContextMenuTarget] = useState<ImageContextMenuTarget | null>(null);
  const onImagesDeleted = useCallback(
    (imageNames: string[]) => {
      const deletedNames = new Set(imageNames);
      const images = boardImagesRef.current;
      const anchorName = selectedImageRef.current?.imageName ?? null;

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
        dispatch({ image: nextImage, type: 'selectGalleryImage' });
      }
    },
    [dispatch]
  );
  const projectId = useActiveProjectId();
  const imageActions = useImageActions({
    boards,
    dispatch,
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
  const isComparing =
    selectedImage !== null && compareImage !== null && compareImage.imageName !== selectedImage.imageName;
  // The reveal window suppresses live frames briefly after a completion so the
  // finished result is seen before the next item's noise takes over.
  const isRevealHolding = useRevealHold();
  const shouldShowProgressImage =
    showProgressImagesInViewer && progressImage !== null && !isComparing && !isRevealHolding;
  const exitCompare = useCallback(() => dispatch({ image: null, type: 'setGalleryCompareImage' }), [dispatch]);
  const swapCompareImages = useCallback(() => {
    if (selectedImage && compareImage) {
      dispatch({ image: compareImage, type: 'selectGalleryImage' });
      dispatch({ image: selectedImage, type: 'setGalleryCompareImage' });
    }
  }, [compareImage, dispatch, selectedImage]);
  const setComparisonMode = useCallback(
    (comparisonMode: PreviewComparisonMode) =>
      dispatch({ type: 'patchWidgetValues', values: { comparisonMode }, widgetId: 'preview' }),
    [dispatch]
  );
  const isMetadataOpen = getPreviewMetadataOpen(previewValues);
  const toggleMetadata = useCallback(
    () => dispatch({ type: 'patchWidgetValues', values: { metadataOpen: !isMetadataOpen }, widgetId: 'preview' }),
    [dispatch, isMetadataOpen]
  );
  const isFilmstripVisible = getPreviewFilmstripVisible(previewValues);
  const selectImage = useCallback(
    (image: GeneratedImageContract) => dispatch({ image, type: 'selectGalleryImage' }),
    [dispatch]
  );

  // Drop-to-compare: any gallery-image drag dropped on the frame's drop zone
  // arms that image for comparison. The drag payload only carries names, so
  // the full contract is fetched before dispatching.
  const handleCompareDrop = useEffectEvent((event: DragEndEvent) => {
    const resolution = resolvePreviewCompareDrop(event.active.data.current, event.over?.data.current ?? null);

    if (!resolution) {
      return;
    }

    // Prefer images we already hold (board context includes fresh local
    // generations that would 404 on a backend by-name fetch).
    const localImage =
      boardImagesRef.current.find((image) => image.imageName === resolution.imageName) ??
      (selectedImageRef.current?.imageName === resolution.imageName ? selectedImageRef.current : null);

    if (localImage) {
      dispatch({ image: localImage, type: 'setGalleryCompareImage' });
      return;
    }

    getGalleryImageByName(resolution.imageName)
      .then((image) => dispatch({ image, type: 'setGalleryCompareImage' }))
      .catch((error: unknown) => {
        dispatch({
          area: 'preview-compare-drop',
          message: error instanceof Error ? error.message : String(error),
          namespace: 'gallery',
          type: 'recordError',
        });
      });
  });
  const onCompareDragEnd = useCallback((event: DragEndEvent) => handleCompareDrop(event), []);

  useDndMonitor({ onDragEnd: onCompareDragEnd });
  const openImageContextMenu = useCallback(
    (x: number, y: number) => {
      if (contextMenuImage) {
        setContextMenuTarget({ images: [contextMenuImage], x, y });
      }
    },
    [contextMenuImage]
  );
  const selectNextImage = useCallback(() => selectByOffset(1), [selectByOffset]);
  const selectPreviousImage = useCallback(() => selectByOffset(-1), [selectByOffset]);
  const closeContextMenu = useCallback(() => setContextMenuTarget(null), []);
  const selectedImageName = selectedImage?.imageName ?? null;

  // Publish the header chrome context (the "[board] / [image]" label and the
  // action strip's image + actions) for the widget frame; the chrome renders
  // outside this view, so an external store is the sync channel. Cleared on
  // unmount so stale chrome never outlives us.
  useEffect(() => {
    previewHeaderStore.set({
      actionImage: contextMenuImage,
      actions: imageActions,
      boardName: selectedImageName === null ? null : boardName,
      imageName: selectedImageName,
      openImageMenu: openImageContextMenu,
    });
  }, [boardName, contextMenuImage, imageActions, openImageContextMenu, selectedImageName]);

  useEffect(() => () => previewHeaderStore.clear(), []);

  // Warm the browser cache for the board neighbors so arrow-key navigation
  // swaps without a decode flash.
  const previousNeighborUrl = boardImages[selectedIndex - 1]?.imageUrl ?? null;
  const nextNeighborUrl = boardImages[selectedIndex + 1]?.imageUrl ?? null;

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
      dispatch({ image: compareImage, type: 'selectGalleryImage' });
      dispatch({ image: selectedImage, type: 'setGalleryCompareImage' });
      return;
    }

    if (commandId === 'viewer.deleteImage' && selectedImage) {
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
      dispatch({
        type: 'patchWidgetValues',
        values: { filmstripVisible: !getPreviewFilmstripVisible(previewValues) },
        widgetId: 'preview',
      });
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
      {selectedImage ? (
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
              boardImageCount={boardImages.length}
              density={density}
              image={selectedImage}
              isLoadingBoard={isLoadingBoard}
              isMetadataOpen={isMetadataOpen}
              progressImage={shouldShowProgressImage ? progressImage : null}
              selectedIndex={selectedIndex}
              shouldAntialiasProgressImage={antialiasProgressImages}
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
        <EmptyPreview
          progressImage={shouldShowProgressImage ? progressImage : null}
          shouldAntialiasProgressImage={antialiasProgressImages}
        />
      )}
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
  progressImage,
  selectedIndex,
  shouldAntialiasProgressImage,
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
  progressImage: LatestProgressImageSnapshot | null;
  selectedIndex: number;
  shouldAntialiasProgressImage: boolean;
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
    liveImage: progressImageToStreamingSource(progressImage),
  });
  const isProgressImage = previewImage?.kind === 'live';
  const previewWidth = previewImage?.width ?? image.width;
  const previewHeight = previewImage?.height ?? image.height;
  const liveQueueItemId = progressImage?.target?.queueItemId ?? null;
  const dragImage = useMemo(
    () => ({ boardId: actionImage?.boardId ?? 'none', imageName: image.imageName }),
    [actionImage?.boardId, image.imageName]
  );
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        onPrevious();
      }

      if (event.key === 'ArrowRight') {
        event.preventDefault();
        onNext();
      }
    },
    [onNext, onPrevious]
  );

  return (
    <Stack gap="3" h="full" minH="0" outline="none" tabIndex={0} w="full" onKeyDown={handleKeyDown}>
      <PreviewFrame
        dragImage={dragImage}
        frameHeight={previewHeight}
        frameWidth={previewWidth}
        isLive={isProgressImage}
        liveBadgeLabel={t('common.generating')}
        liveQueueItemId={liveQueueItemId}
        loupeControlsRef={loupeControlsRef}
        padding={density === 'full' ? '6' : '3'}
        shouldAntialiasLiveImage={shouldAntialiasProgressImage}
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
        isLive={isProgressImage}
        isLoadingBoard={isLoadingBoard}
        isMetadataOpen={isMetadataOpen}
        liveQueueItemId={liveQueueItemId}
        selectedIndex={selectedIndex}
        onNext={onNext}
        onPrevious={onPrevious}
        onToggleMetadata={onToggleMetadata}
      />
    </Stack>
  );
};

const EmptyPreview = ({
  progressImage,
  shouldAntialiasProgressImage,
}: {
  progressImage: LatestProgressImageSnapshot | null;
  shouldAntialiasProgressImage: boolean;
}) => {
  const { t } = useTranslation();
  const previewImage = useStreamingImageSource({
    liveImage: progressImageToStreamingSource(progressImage),
  });

  return (
    <PreviewFrame
      frameHeight={previewImage?.height ?? 1}
      frameWidth={previewImage?.width ?? 1}
      isLive
      liveBadgeLabel={t('common.generating')}
      liveQueueItemId={progressImage?.target?.queueItemId ?? null}
      shouldAntialiasLiveImage={shouldAntialiasProgressImage}
      source={previewImage}
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

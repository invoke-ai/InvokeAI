/* eslint-disable react/react-compiler */
import type { GeneratedImageContract, WidgetViewProps } from '@workbench/types';

import { Box, Stack, Text } from '@chakra-ui/react';
import { useProgressImage, type ProgressImageSnapshot } from '@workbench/backend/progressImageStore';
import {
  listGalleryBoards,
  listGalleryImages,
  type GalleryBoard,
  type GalleryImage,
  type GalleryView,
} from '@workbench/gallery/api';
import { getGallerySettings } from '@workbench/gallery/settings';
import { ImageContextMenu, useImageActions, type ImageContextMenuTarget } from '@workbench/image-actions';
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
import { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { PreviewCompare } from './PreviewCompare';
import { usePreviewDensity, type PreviewDensity } from './previewDensity';
import { PreviewFooter } from './PreviewFooter';
import { PreviewFrame } from './PreviewFrame';
import { getPreviewComparisonMode, type PreviewComparisonMode } from './previewSettings';

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
  const shouldShowProgressImage = showProgressImagesInViewer && progressImage !== null && !isComparing;
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
    }
  });

  useEffect(() => {
    const hotkeys = [
      ['viewer.toggleViewer', t('widgets.preview.commands.togglePreview'), ['z']],
      ['viewer.swapImages', t('widgets.preview.commands.swapComparisonImages'), ['c']],
      ['viewer.deleteImage', t('widgets.preview.commands.deletePreviewImage'), ['delete', 'backspace']],
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
              boardImageCount={boardImages.length}
              boardName={boardName}
              density={density}
              image={selectedImage}
              isLoadingBoard={isLoadingBoard}
              progressImage={shouldShowProgressImage ? progressImage : null}
              selectedIndex={selectedIndex}
              shouldAntialiasProgressImage={antialiasProgressImages}
              onContextMenu={openImageContextMenu}
              onNext={selectNextImage}
              onPrevious={selectPreviousImage}
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
  boardImageCount,
  boardName,
  density,
  image,
  isLoadingBoard,
  progressImage,
  selectedIndex,
  shouldAntialiasProgressImage,
  onContextMenu,
  onNext,
  onPrevious,
}: {
  boardImageCount: number;
  boardName: string;
  density: PreviewDensity;
  image: GeneratedImageContract;
  isLoadingBoard: boolean;
  progressImage: ProgressImageSnapshot | null;
  selectedIndex: number;
  shouldAntialiasProgressImage: boolean;
  onContextMenu: (x: number, y: number) => void;
  onNext: () => void;
  onPrevious: () => void;
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
        frameHeight={previewHeight}
        frameWidth={previewWidth}
        isLive={isProgressImage}
        liveBadgeLabel={t('common.generating')}
        padding={density === 'full' ? '6' : '3'}
        shouldAntialiasLiveImage={shouldAntialiasProgressImage}
        source={previewImage}
        variant="framed"
        onContextMenu={onContextMenu}
      />
      <PreviewFooter
        boardImageCount={boardImageCount}
        boardName={boardName}
        image={image}
        isLive={isProgressImage}
        isLoadingBoard={isLoadingBoard}
        selectedIndex={selectedIndex}
        onNext={onNext}
        onPrevious={onPrevious}
      />
    </Stack>
  );
};

const EmptyPreview = ({
  progressImage,
  shouldAntialiasProgressImage,
}: {
  progressImage: ProgressImageSnapshot | null;
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

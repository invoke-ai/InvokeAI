/* eslint-disable react/react-compiler */
import type { GeneratedImageContract, WidgetViewProps } from '@workbench/types';

import { Badge, Box, Flex, HStack, Stack, Text } from '@chakra-ui/react';
import { useProgressImage, type ProgressImageSnapshot } from '@workbench/backend/progressImageStore';
import { Button } from '@workbench/components/ui';
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
import { ChevronLeftIcon, ChevronRightIcon } from 'lucide-react';
import {
  useCallback,
  useEffect,
  useEffectEvent,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type CSSProperties,
} from 'react';

import { PreviewCompare } from './PreviewCompare';

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

const getBoardName = (boards: GalleryBoard[], boardId: string): string =>
  boards.find((board) => board.id === boardId)?.name ?? (boardId === 'none' ? 'Uncategorized' : 'Unknown board');

const shouldLoadBackendBoard = (image: PreviewImage): boolean => image.sourceQueueItemId === 'backend-gallery';

export const getPreviewBoardsRequestKey = ({
  hasSelectedImage,
  refreshToken,
}: {
  hasSelectedImage: boolean;
  isBackendImage: boolean;
  refreshToken: string;
}): string | null => (hasSelectedImage ? refreshToken : null);

const previewGridCss = {
  backgroundImage: 'radial-gradient(circle, currentColor 1px, transparent 1.5px)',
  backgroundPosition: 'center',
  backgroundRepeat: 'repeat',
  backgroundSize: '24px 24px',
} as const;

const selectGenerateRecallValues = createGenerateFormValuesSelector();

export const PreviewWidgetView = ({ region, runtime }: WidgetViewProps) => {
  const galleryValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'gallery'));
  const generateValues = useWidgetValuesSelector('generate', selectGenerateRecallValues);
  const { antialiasProgressImages, showProgressImagesInViewer } = useActiveProjectSelector(
    (project) => project.settings
  );
  const progressImage = useProgressImage();
  const dispatch = useWorkbenchDispatch();
  const selectedImage = useMemo(() => getSelectedImage(galleryValues), [galleryValues]);
  const compareImage = getGalleryCompareImage(galleryValues);
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
  const isSidePanel = region === 'left' || region === 'right';
  const selectedIndex = selectedImage
    ? boardImages.findIndex((image) => image.imageName === selectedImage.imageName)
    : -1;
  const boardName = getBoardName(boards, displayBoardId);
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
          dispatch({ message: error instanceof Error ? error.message : String(error), type: 'recordError' });

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
      ['viewer.toggleViewer', 'Toggle preview', ['z']],
      ['viewer.swapImages', 'Swap comparison images', ['c']],
      ['viewer.deleteImage', 'Delete preview image', ['delete', 'backspace']],
    ] as const;
    const disposers = hotkeys.flatMap(([id, title, defaultKeys]) => [
      runtime.commands.register({ handler: () => executeViewerHotkey(id), id, title }),
      runtime.hotkeys.register({ commandId: id, defaultKeys: [...defaultKeys], id, title }),
    ]);

    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [runtime.commands, runtime.hotkeys]);

  return (
    <Box p="2" h="full">
      {selectedImage ? (
        <>
          {isComparing && compareImage ? (
            <PreviewCompare
              baseImage={selectedImage}
              compareImage={compareImage}
              runtime={runtime}
              onExit={exitCompare}
              onSwap={swapCompareImages}
            />
          ) : (
            <SelectedImagePreview
              boardImageCount={boardImages.length}
              boardName={boardName}
              image={selectedImage}
              isCompact={isSidePanel}
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
  image,
  isCompact,
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
  image: GeneratedImageContract;
  isCompact: boolean;
  isLoadingBoard: boolean;
  progressImage: ProgressImageSnapshot | null;
  selectedIndex: number;
  shouldAntialiasProgressImage: boolean;
  onContextMenu: (x: number, y: number) => void;
  onNext: () => void;
  onPrevious: () => void;
}) => {
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
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      event.preventDefault();
      onContextMenu(event.clientX, event.clientY);
    },
    [onContextMenu]
  );
  const imageStyle = useMemo<CSSProperties>(
    () => ({
      display: 'block',
      height: '100%',
      imageRendering: isProgressImage && !shouldAntialiasProgressImage ? 'pixelated' : 'auto',
      inset: 0,
      objectFit: 'contain',
      position: 'absolute',
      width: '100%',
    }),
    [isProgressImage, shouldAntialiasProgressImage]
  );

  return (
    <Stack gap="3" h="full" minH="0" tabIndex={0} w="full" onKeyDown={handleKeyDown}>
      <Flex
        align="center"
        borderWidth="1px"
        borderColor="border.subtle"
        color="fg.grid"
        css={previewGridCss}
        flex="1"
        justify="center"
        minH="0"
        overflow="hidden"
        p={isCompact ? '3' : '6'}
        rounded="lg"
        w="full"
      >
        <Box
          aspectRatio={previewWidth / previewHeight}
          bg="transparent"
          borderWidth="1px"
          borderColor={isProgressImage ? 'accent.solid' : 'border.emphasized'}
          boxShadow="0 24px 80px rgba(0,0,0,0.42)"
          h={isCompact ? 'auto' : 'full'}
          maxH={isCompact ? undefined : 'full'}
          maxW="full"
          overflow="hidden"
          position="relative"
          rounded="lg"
          w={isCompact ? 'full' : 'auto'}
          onContextMenu={handleContextMenu}
        >
          {previewImage ? <img alt={previewImage.alt} src={previewImage.src} style={imageStyle} /> : null}
          {isProgressImage ? (
            <Badge left="2" pointerEvents="none" position="absolute" size="xs" top="2" variant="solid">
              Generating
            </Badge>
          ) : null}
        </Box>
      </Flex>
      <Stack borderWidth="1px" borderColor="border.subtle" gap="2" p="3" rounded="lg">
        <HStack align="center" justify="space-between">
          <HStack gap="2" minW="0">
            <Badge flexShrink={0} size="xs" variant="subtle">
              {boardName}
            </Badge>
            <Text color="fg.subtle" fontSize="2xs" truncate>
              {isLoadingBoard
                ? 'Loading board'
                : selectedIndex === -1
                  ? `${boardImageCount} image${boardImageCount === 1 ? '' : 's'}`
                  : `${selectedIndex + 1} of ${boardImageCount}`}
            </Text>
          </HStack>
          <HStack flexShrink={0} gap="1">
            <Button
              aria-label="Previous image in board"
              disabled={selectedIndex <= 0}
              size="2xs"
              variant="outline"
              onClick={onPrevious}
            >
              <ChevronLeftIcon />
            </Button>
            <Button
              aria-label="Next image in board"
              disabled={selectedIndex === -1 || selectedIndex >= boardImageCount - 1}
              size="2xs"
              variant="outline"
              onClick={onNext}
            >
              <ChevronRightIcon />
            </Button>
          </HStack>
        </HStack>
        <Stack>
          <Text fontSize="xs" fontWeight="800" truncate>
            {image.imageName}
          </Text>
          <Text color="fg.subtle" fontSize="2xs">
            Source run {image.sourceQueueItemId}
          </Text>
          <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
            {isProgressImage ? 'Generating' : `${image.width} x ${image.height}`}
          </Text>
        </Stack>
      </Stack>
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
  const previewImage = useStreamingImageSource({
    liveImage: progressImageToStreamingSource(progressImage),
  });

  return (
    <Flex
      align="center"
      backgroundColor="bg.inset"
      color="fg.grid"
      css={previewGridCss}
      h="full"
      justify="center"
      w="full"
    >
      {previewImage ? (
        <Box
          aspectRatio={(previewImage.width ?? 1) / (previewImage.height ?? 1)}
          borderColor="accent.solid"
          borderWidth="1px"
          boxShadow="0 24px 80px rgba(0,0,0,0.42)"
          maxH="full"
          maxW="full"
          overflow="hidden"
          position="relative"
          rounded="lg"
        >
          <img
            alt={previewImage.alt}
            draggable={false}
            src={previewImage.src}
            style={{
              display: 'block',
              height: '100%',
              imageRendering: shouldAntialiasProgressImage ? 'auto' : 'pixelated',
              objectFit: 'contain',
              width: '100%',
            }}
          />
          <Badge left="2" pointerEvents="none" position="absolute" size="xs" top="2" variant="solid">
            Generating
          </Badge>
        </Box>
      ) : (
        <Stack align="center" color="fg" gap="2" maxW="18rem" textAlign="center">
          <Text fontSize="sm" fontWeight="800">
            No gallery selection
          </Text>
          <Text color="fg.subtle" fontSize="2xs">
            Generate to Gallery, then select a thumbnail in the Gallery widget to inspect it here.
          </Text>
        </Stack>
      )}
    </Flex>
  );
};

import { Badge, Box, Button, Flex, HStack, Stack, Text } from '@chakra-ui/react';
import { useEffect, useMemo, useState } from 'react';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

import {
  listGalleryBoards,
  listGalleryImages,
  type GalleryBoard,
  type GalleryImage,
  type GalleryView,
} from '../../gallery/api';
import type { GeneratedImageContract, WidgetViewProps } from '../../types';
import { useWorkbench } from '../../WorkbenchContext';

type PreviewImage = GeneratedImageContract & Partial<Pick<GalleryImage, 'boardId' | 'imageCategory'>>;

const getImageAspectRatio = (image: GeneratedImageContract): number => image.width / image.height;

const fallbackBoards: GalleryBoard[] = [
  { assetCount: 0, id: 'none', imageCount: 0, isVirtual: true, name: 'Uncategorized' },
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

const getImageBoardId = (image: PreviewImage, values: Record<string, unknown>): string => {
  if (typeof image.boardId === 'string') {
    return image.boardId;
  }

  const selectedBoardId = typeof values.selectedBoardId === 'string' ? values.selectedBoardId : 'none';

  return selectedBoardId;
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

export const PreviewWidgetView = ({ region }: WidgetViewProps) => {
  const { activeProject, dispatch } = useWorkbench();
  const galleryValues = activeProject.widgetStates.gallery.values;
  const selectedImage = useMemo(() => getSelectedImage(galleryValues), [galleryValues]);
  const selectedBoardId = selectedImage ? getImageBoardId(selectedImage, galleryValues) : 'none';
  const selectedGalleryView = selectedImage ? getImageGalleryView(selectedImage) : 'images';
  const localImages = useMemo(
    () => getGalleryImages(galleryValues).filter((image) => image.boardId === selectedBoardId),
    [galleryValues, selectedBoardId]
  );
  const [boards, setBoards] = useState<GalleryBoard[]>(fallbackBoards);
  const [boardImages, setBoardImages] = useState<PreviewImage[]>(localImages);
  const [isLoadingBoard, setIsLoadingBoard] = useState(false);
  const isSidePanel = region === 'left' || region === 'right';
  const selectedIndex = selectedImage
    ? boardImages.findIndex((image) => image.imageName === selectedImage.imageName)
    : -1;
  const boardName = getBoardName(boards, selectedBoardId);

  useEffect(() => {
    if (!selectedImage) {
      setBoardImages([]);
      setIsLoadingBoard(false);
      return;
    }

    if (!shouldLoadBackendBoard(selectedImage)) {
      setBoardImages(localImages);
      setIsLoadingBoard(false);
      return;
    }

    let isStale = false;

    setIsLoadingBoard(true);
    Promise.all([
      listGalleryBoards().catch(() => fallbackBoards),
      listGalleryImages({ boardId: selectedBoardId, galleryView: selectedGalleryView, searchTerm: '' }).catch(() => []),
    ])
      .then(([nextBoards, nextImages]) => {
        if (isStale) {
          return;
        }

        setBoards(nextBoards);
        setBoardImages(nextImages.length ? nextImages : [selectedImage]);
      })
      .catch((error: unknown) => {
        if (!isStale) {
          dispatch({ message: error instanceof Error ? error.message : String(error), type: 'recordError' });
          setBoardImages([selectedImage]);
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
  }, [dispatch, localImages, selectedBoardId, selectedGalleryView, selectedImage]);

  const selectByOffset = (offset: -1 | 1) => {
    if (selectedIndex === -1) {
      return;
    }

    const nextImage = boardImages[selectedIndex + offset];

    if (nextImage) {
      dispatch({ image: nextImage, type: 'selectGalleryImage' });
    }
  };

  return selectedImage ? (
    <SelectedImagePreview
      boardImageCount={boardImages.length}
      boardName={boardName}
      image={selectedImage}
      isCompact={isSidePanel}
      isLoadingBoard={isLoadingBoard}
      selectedIndex={selectedIndex}
      onNext={() => selectByOffset(1)}
      onPrevious={() => selectByOffset(-1)}
    />
  ) : (
    <EmptyPreview />
  );
};

const SelectedImagePreview = ({
  boardImageCount,
  boardName,
  image,
  isCompact,
  isLoadingBoard,
  selectedIndex,
  onNext,
  onPrevious,
}: {
  boardImageCount: number;
  boardName: string;
  image: GeneratedImageContract;
  isCompact: boolean;
  isLoadingBoard: boolean;
  selectedIndex: number;
  onNext: () => void;
  onPrevious: () => void;
}) => (
  <Stack
    gap="3"
    h="full"
    minH="0"
    tabIndex={0}
    w="full"
    onKeyDown={(event) => {
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        onPrevious();
      }

      if (event.key === 'ArrowRight') {
        event.preventDefault();
        onNext();
      }
    }}
  >
    <Flex align="center" flex="1" justify="center" minH="0" w="full">
      <Box
        aspectRatio={getImageAspectRatio(image)}
        bg="bg.panel"
        borderWidth="1px"
        borderColor="border.emphasis"
        boxShadow="0 24px 80px rgba(0,0,0,0.42)"
        h={isCompact ? 'auto' : 'full'}
        maxH={isCompact ? undefined : 'full'}
        maxW="full"
        overflow="hidden"
        position="relative"
        rounded="lg"
        w={isCompact ? 'full' : 'auto'}
      >
        <img
          alt={image.imageName}
          src={image.imageUrl}
          style={{
            display: 'block',
            height: '100%',
            inset: 0,
            objectFit: 'contain',
            position: 'absolute',
            width: '100%',
          }}
        />
      </Box>
    </Flex>
    <Stack bg="bg.surfaceRaised" borderWidth="1px" borderColor="border.subtle" gap="2" p="3" rounded="lg">
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
            <PiCaretLeftBold />
          </Button>
          <Button
            aria-label="Next image in board"
            disabled={selectedIndex === -1 || selectedIndex >= boardImageCount - 1}
            size="2xs"
            variant="outline"
            onClick={onNext}
          >
            <PiCaretRightBold />
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
          {image.width} x {image.height}
        </Text>
      </Stack>
    </Stack>
  </Stack>
);

const EmptyPreview = () => (
  <Flex align="center" h="full" justify="center" w="full">
    <Stack align="center" gap="2" maxW="18rem" textAlign="center">
      <Text fontSize="sm" fontWeight="800">
        No gallery selection
      </Text>
      <Text color="fg.subtle" fontSize="2xs">
        Generate to Gallery, then select a thumbnail in the Gallery widget to inspect it here.
      </Text>
    </Stack>
  </Flex>
);

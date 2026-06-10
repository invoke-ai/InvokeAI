import {
  Badge,
  Box,
  Button,
  Flex,
  HStack,
  Icon,
  IconButton,
  Image,
  Input,
  Menu,
  Portal,
  Slider,
  Stack,
  Text,
} from '@chakra-ui/react';
import { useState } from 'react';
import { PiCaretDownBold, PiGearSixBold, PiImageBold, PiPlusBold } from 'react-icons/pi';

import type { GalleryBoard, GalleryView } from '../../gallery/api';
import type { GeneratedImageContract, WorkbenchRegion } from '../../types';
import { getBoardCounts, type GalleryStateView } from './galleryStateView';

export const GalleryPanelContent = ({
  gallery,
  layout,
  region,
  onCreateBoard,
  onSearch,
  onSetImageDensityPercent,
  onSelectBoard,
  onSelectImage,
  onSetView,
}: {
  gallery: GalleryStateView;
  layout: 'stacked' | 'wide';
  region: WorkbenchRegion;
  onCreateBoard: () => void;
  onSearch: (searchTerm: string) => void;
  onSetImageDensityPercent: (imageDensityPercent: number) => void;
  onSelectBoard: (boardId: string) => void;
  onSelectImage: (image: GeneratedImageContract) => void;
  onSetView: (galleryView: GalleryView) => void;
}) => {
  const isWide = layout === 'wide';

  return (
    <Flex direction={isWide ? 'row' : 'column'} gap="3" h="full" maxW="full" minH="0" minW="0" w="full">
      <GalleryBoardSelect
        gallery={gallery}
        layout={layout}
        onCreateBoard={onCreateBoard}
        onSelectBoard={onSelectBoard}
      />
      <Stack flex="1" gap="3" maxW="full" minH="0" minW="0" w="full">
        <GalleryToolbar
          gallery={gallery}
          onSearch={onSearch}
          onSetImageDensityPercent={onSetImageDensityPercent}
          onSetView={onSetView}
        />
        <GalleryImages
          columnCount={getGalleryColumnCount(gallery.imageDensityPercent, layout)}
          imageDensityPercent={gallery.imageDensityPercent}
          images={gallery.images}
          isLoading={gallery.isLoading}
          region={region}
          selectedImageName={gallery.selectedImageName}
          onNavigate={(direction) => {
            const selectedIndex = gallery.images.findIndex((image) => image.imageName === gallery.selectedImageName);
            const fallbackIndex = selectedIndex === -1 ? 0 : selectedIndex;
            const nextIndex =
              direction === 'next'
                ? Math.min(gallery.images.length - 1, fallbackIndex + 1)
                : Math.max(0, fallbackIndex - 1);
            const nextImage = gallery.images[nextIndex];

            if (nextImage) {
              onSelectImage(nextImage);
            }
          }}
          onSelect={onSelectImage}
        />
      </Stack>
    </Flex>
  );
};

const GalleryBoardSelect = ({
  gallery,
  layout,
  onCreateBoard,
  onSelectBoard,
}: {
  gallery: GalleryStateView;
  layout: 'stacked' | 'wide';
  onCreateBoard: () => void;
  onSelectBoard: (boardId: string) => void;
}) => {
  const isWide = layout === 'wide';
  const [boardSearchTerm, setBoardSearchTerm] = useState('');
  const normalizedSearchTerm = boardSearchTerm.trim().toLowerCase();
  const selectedBoard = gallery.boards.find((board) => board.id === gallery.selectedBoardId) ?? null;
  const filteredBoards = normalizedSearchTerm
    ? gallery.boards.filter((board) => board.name.toLowerCase().includes(normalizedSearchTerm))
    : gallery.boards;

  return (
    <Stack flex={isWide ? '0 0 16rem' : '0 0 auto'} gap="2" maxW="full" minW="0" w={isWide ? undefined : 'full'}>
      <HStack justify="space-between">
        <Text fontSize="xs" fontWeight="700">
          Boards
        </Text>
        <Button size="2xs" variant="outline" onClick={onCreateBoard}>
          <PiPlusBold />
          Board
        </Button>
      </HStack>
      <Menu.Root positioning={{ placement: isWide ? 'bottom-start' : 'bottom' }}>
        <Menu.Trigger asChild>
          <Button justifyContent="space-between" size="sm" variant="outline" w="full" px="1">
            {selectedBoard ? (
              <BoardOptionContent board={selectedBoard} isSelected={false} />
            ) : (
              <Text>Loading board...</Text>
            )}
            <Icon as={PiCaretDownBold} boxSize="3" flexShrink={0} />
          </Button>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <Menu.Content
              bg="bg.surfaceRaised"
              borderWidth="1px"
              borderColor="border.emphasis"
              color="fg.default"
              maxH="min(28rem, calc(100vh - 6rem))"
              minW={isWide ? '16rem' : 'min(22rem, calc(100vw - 2rem))'}
              overflow="hidden"
              rounded="lg"
              shadow="lg"
            >
              <Box p="2">
                <Input
                  aria-label="Search boards"
                  placeholder="Search boards"
                  size="sm"
                  value={boardSearchTerm}
                  onChange={(event) => setBoardSearchTerm(event.currentTarget.value)}
                  onKeyDown={(event) => event.stopPropagation()}
                />
              </Box>
              <Menu.Separator borderColor="border.subtle" />
              <Box maxH="18rem" overflowY="auto" py="1">
                {filteredBoards.length ? (
                  filteredBoards.map((board) => {
                    const isSelected = board.id === gallery.selectedBoardId;

                    return (
                      <Menu.Item key={board.id} value={board.id} onClick={() => onSelectBoard(board.id)}>
                        <BoardOptionContent board={board} isSelected={isSelected} />
                      </Menu.Item>
                    );
                  })
                ) : (
                  <Text color="fg.subtle" fontSize="2xs" px="3" py="2">
                    No boards match this search.
                  </Text>
                )}
              </Box>
            </Menu.Content>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </Stack>
  );
};

const BoardOptionContent = ({ board, isSelected }: { board: GalleryBoard; isSelected: boolean }) => {
  const counts = getBoardCounts(board);

  return (
    <HStack gap="2" minW="0" w="full">
      <BoardCover board={board} />
      <Text flex="1" fontSize="xs" fontWeight={isSelected ? '700' : '500'} minW="0" truncate>
        {board.name}
      </Text>
      <Badge flexShrink={0} size="xs" variant={isSelected ? 'solid' : 'subtle'}>
        {counts.imageCount} | {counts.assetCount}
      </Badge>
    </HStack>
  );
};

const BoardCover = ({ board }: { board: GalleryBoard }) => {
  if (board.coverThumbnailUrl) {
    return (
      <Image
        alt=""
        bg="bg.panel"
        borderWidth="1px"
        borderColor="border.subtle"
        boxSize="7"
        flexShrink={0}
        objectFit="cover"
        rounded="md"
        src={board.coverThumbnailUrl}
      />
    );
  }

  return (
    <Flex
      align="center"
      bg="bg.panel"
      borderWidth="1px"
      borderColor="border.subtle"
      boxSize="7"
      color="fg.subtle"
      flexShrink={0}
      justify="center"
      rounded="md"
    >
      <Icon as={PiImageBold} boxSize="4" />
    </Flex>
  );
};

const GalleryToolbar = ({
  gallery,
  onSearch,
  onSetImageDensityPercent,
  onSetView,
}: {
  gallery: GalleryStateView;
  onSearch: (searchTerm: string) => void;
  onSetImageDensityPercent: (imageDensityPercent: number) => void;
  onSetView: (galleryView: GalleryView) => void;
}) => (
  <Stack gap="2">
    <HStack justify="space-between">
      <HStack gap="1">
        <Button
          size="xs"
          variant={gallery.galleryView === 'images' ? 'solid' : 'outline'}
          onClick={() => onSetView('images')}
        >
          Images
        </Button>
        <Button
          size="xs"
          variant={gallery.galleryView === 'assets' ? 'solid' : 'outline'}
          onClick={() => onSetView('assets')}
        >
          Assets
        </Button>
      </HStack>
      <HStack gap="2">
        <Text color="fg.subtle" fontSize="2xs">
          {gallery.images.length} total
        </Text>
        <GallerySettingsMenu gallery={gallery} onSetImageDensityPercent={onSetImageDensityPercent} />
      </HStack>
    </HStack>
    <Input
      aria-label="Search gallery images"
      placeholder="Search images"
      size="sm"
      value={gallery.searchTerm}
      onChange={(event) => onSearch(event.currentTarget.value)}
    />
  </Stack>
);

const GallerySettingsMenu = ({
  gallery,
  onSetImageDensityPercent,
}: {
  gallery: GalleryStateView;
  onSetImageDensityPercent: (imageDensityPercent: number) => void;
}) => (
  <Menu.Root positioning={{ placement: 'bottom-end' }}>
    <Menu.Trigger asChild>
      <IconButton aria-label="Gallery settings" size="xs" variant="outline">
        <PiGearSixBold />
      </IconButton>
    </Menu.Trigger>
    <Portal>
      <Menu.Positioner>
        <Menu.Content
          bg="bg.surfaceRaised"
          borderWidth="1px"
          borderColor="border.emphasis"
          color="fg.default"
          minW="14rem"
          p="3"
          rounded="lg"
          shadow="lg"
        >
          <Stack gap="2">
            <HStack justify="space-between">
              <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
                Grid Density
              </Text>
              <Text color="fg.subtle" fontSize="2xs">
                {gallery.imageDensityPercent}%
              </Text>
            </HStack>
            <Slider.Root
              aria-label={['Gallery image density']}
              max={100}
              min={0}
              step={1}
              value={[gallery.imageDensityPercent]}
              onKeyDown={(event) => event.stopPropagation()}
              onValueChange={(event) => onSetImageDensityPercent(event.value[0] ?? gallery.imageDensityPercent)}
            >
              <Slider.Control>
                <Slider.Track>
                  <Slider.Range />
                </Slider.Track>
                <Slider.Thumbs />
              </Slider.Control>
            </Slider.Root>
          </Stack>
        </Menu.Content>
      </Menu.Positioner>
    </Portal>
  </Menu.Root>
);

const getGalleryColumnCount = (imageDensityPercent: number, layout: 'stacked' | 'wide'): number => {
  const min = 2;
  const max = layout === 'wide' ? 10 : 5;
  const percent = Math.min(100, Math.max(0, imageDensityPercent));

  return Math.round(min + ((max - min) * percent) / 100);
};

const getCompactThumbnailSizeRem = (imageDensityPercent: number): number => {
  const percent = Math.min(100, Math.max(0, imageDensityPercent));

  return 7 - (3 * percent) / 100;
};

const GalleryImages = ({
  columnCount,
  imageDensityPercent,
  images,
  isLoading,
  region,
  selectedImageName,
  onNavigate,
  onSelect,
}: {
  columnCount: number;
  imageDensityPercent: number;
  images: GeneratedImageContract[];
  isLoading: boolean;
  region: WorkbenchRegion;
  selectedImageName: string | null;
  onNavigate: (direction: 'next' | 'previous') => void;
  onSelect: (image: GeneratedImageContract) => void;
}) => {
  const isLargePlacement = region === 'center' || region === 'bottom';
  const thumbnailAspectRatio = isLargePlacement ? 1 : 4 / 3;
  const selectedIndex = images.findIndex((image) => image.imageName === selectedImageName);
  const compactThumbnailSizeRem = getCompactThumbnailSizeRem(imageDensityPercent);
  const gridTemplateColumns = isLargePlacement
    ? `repeat(${columnCount}, minmax(0, 1fr))`
    : `repeat(auto-fill, minmax(min(100%, ${compactThumbnailSizeRem}rem), ${compactThumbnailSizeRem}rem))`;

  if (images.length === 0) {
    return (
      <Flex align="center" color="fg.subtle" flex="1" justify="center" minH="8rem">
        <Text fontSize="2xs">
          {isLoading ? 'Loading backend gallery...' : 'No images match this board, tab, or search.'}
        </Text>
      </Flex>
    );
  }

  return (
    <Box
      aria-label="Gallery images"
      flex="1"
      maxW="full"
      minH="0"
      minW="0"
      overflowX="hidden"
      overflowY="auto"
      role="listbox"
      tabIndex={0}
      w="full"
      onKeyDown={(event) => {
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
          event.preventDefault();
          onNavigate('next');
        }

        if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
          event.preventDefault();
          onNavigate('previous');
        }
      }}
    >
      <Box
        display="grid"
        gap="2"
        gridTemplateColumns={gridTemplateColumns}
        justifyContent={isLargePlacement ? undefined : 'start'}
        maxW="full"
        minW="0"
        w="full"
      >
        {images.map((image, index) => (
          <GalleryThumbnail
            key={image.imageName}
            image={image}
            index={index}
            isSelected={image.imageName === selectedImageName}
            selectedIndex={selectedIndex}
            thumbnailAspectRatio={thumbnailAspectRatio}
            onSelect={onSelect}
          />
        ))}
      </Box>
    </Box>
  );
};

const GalleryThumbnail = ({
  image,
  index,
  isSelected,
  selectedIndex,
  thumbnailAspectRatio,
  onSelect,
}: {
  image: GeneratedImageContract;
  index: number;
  isSelected: boolean;
  selectedIndex: number;
  thumbnailAspectRatio: number;
  onSelect: (image: GeneratedImageContract) => void;
}) => (
  <Box
    aspectRatio={thumbnailAspectRatio}
    bg="bg.panel"
    borderWidth="2px"
    borderColor={isSelected ? 'accent.active' : 'border.subtle'}
    minW="0"
    overflow="hidden"
    role="option"
    aria-selected={isSelected}
    position="relative"
    rounded="md"
    w="full"
  >
    <button
      aria-label={`Select ${image.imageName} for preview`}
      style={{
        background: 'transparent',
        border: 0,
        cursor: 'pointer',
        display: 'block',
        height: '100%',
        inset: 0,
        minWidth: 0,
        padding: 0,
        position: 'absolute',
        width: '100%',
      }}
      tabIndex={selectedIndex === -1 ? (index === 0 ? 0 : -1) : isSelected ? 0 : -1}
      type="button"
      onClick={() => onSelect(image)}
    >
      <img
        alt={image.imageName}
        src={image.thumbnailUrl || image.imageUrl}
        style={{
          display: 'block',
          height: '100%',
          inset: 0,
          maxWidth: 'none',
          objectFit: 'cover',
          position: 'absolute',
          width: '100%',
        }}
      />
    </button>
  </Box>
);

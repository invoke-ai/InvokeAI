import { Box, Flex, VStack, useDisclosure } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { memo, useRef } from 'react';
import BoardsList from './Boards/BoardsList/BoardsList';
import GalleryBoardName from './GalleryBoardName';
import GalleryPinButton from './GalleryPinButton';
import GallerySettingsPopover from './GallerySettingsPopover';
import BatchImageGrid from './ImageGrid/BatchImageGrid';
import GalleryImageGrid from './ImageGrid/GalleryImageGrid';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const { selectedBoardId } = state.gallery;

    return {
      selectedBoardId,
    };
  },
  defaultSelectorOptions
);

const ImageGalleryContent = () => {
  const resizeObserverRef = useRef<HTMLDivElement>(null);
  const galleryGridRef = useRef<HTMLDivElement>(null);
  const { selectedBoardId } = useAppSelector(selector);
  const { isOpen: isBoardListOpen, onToggle: onToggleBoardList } =
    useDisclosure();

  return (
    <VStack
      sx={{
        flexDirection: 'column',
        h: 'full',
        w: 'full',
        borderRadius: 'base',
      }}
    >
      <Box sx={{ w: 'full' }}>
        <Flex
          ref={resizeObserverRef}
          sx={{
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 2,
          }}
        >
          <GallerySettingsPopover />
          <GalleryBoardName
            isOpen={isBoardListOpen}
            onToggle={onToggleBoardList}
          />
          <GalleryPinButton />
        </Flex>
        <Box>
          <BoardsList isOpen={isBoardListOpen} />
        </Box>
      </Box>
      <Flex ref={galleryGridRef} direction="column" gap={2} h="full" w="full">
        {selectedBoardId === 'batch' ? (
          <BatchImageGrid />
        ) : (
          <GalleryImageGrid />
        )}
      </Flex>
    </VStack>
  );
};

export default memo(ImageGalleryContent);

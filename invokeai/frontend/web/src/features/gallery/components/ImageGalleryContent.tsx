import {
  Box,
  ButtonGroup,
  Flex,
  Spacer,
  Tab,
  TabList,
  Tabs,
  VStack,
  useDisclosure,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { memo, useCallback, useRef } from 'react';
import BoardsList from './Boards/BoardsList/BoardsList';
import GalleryBoardName from './GalleryBoardName';
import GalleryPinButton from './GalleryPinButton';
import GallerySettingsPopover from './GallerySettingsPopover';
import BatchImageGrid from './ImageGrid/BatchImageGrid';
import GalleryImageGrid from './ImageGrid/GalleryImageGrid';
import IAIButton from 'common/components/IAIButton';
import { FaImages, FaServer } from 'react-icons/fa';
import { galleryViewChanged } from '../store/gallerySlice';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const { selectedBoardId, galleryView } = state.gallery;

    return {
      selectedBoardId,
      galleryView,
    };
  },
  defaultSelectorOptions
);

const ImageGalleryContent = () => {
  const resizeObserverRef = useRef<HTMLDivElement>(null);
  const galleryGridRef = useRef<HTMLDivElement>(null);
  const { selectedBoardId, galleryView } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { isOpen: isBoardListOpen, onToggle: onToggleBoardList } =
    useDisclosure();

  const handleClickImages = useCallback(() => {
    dispatch(galleryViewChanged('images'));
  }, [dispatch]);

  const handleClickAssets = useCallback(() => {
    dispatch(galleryViewChanged('assets'));
  }, [dispatch]);

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
          <GalleryBoardName
            isOpen={isBoardListOpen}
            onToggle={onToggleBoardList}
          />
          <GallerySettingsPopover />
          <GalleryPinButton />
        </Flex>
        <Box>
          <BoardsList isOpen={isBoardListOpen} />
        </Box>
      </Box>
      <Flex ref={galleryGridRef} direction="column" gap={2} h="full" w="full">
        <Flex
          sx={{
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 2,
          }}
        >
          <Tabs
            index={galleryView === 'images' ? 0 : 1}
            variant="line"
            size="sm"
            sx={{ w: 'full' }}
          >
            <TabList>
              <Tab
                onClick={handleClickImages}
                sx={{ borderTopRadius: 'base', w: 'full' }}
              >
                Images
              </Tab>
              <Tab
                onClick={handleClickAssets}
                sx={{ borderTopRadius: 'base', w: 'full' }}
              >
                Assets
              </Tab>
            </TabList>
          </Tabs>
        </Flex>

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

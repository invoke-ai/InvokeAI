import {
  Box,
  ButtonGroup,
  Flex,
  Tab,
  TabList,
  Tabs,
  VStack,
  useDisclosure,
} from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { galleryViewChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaImages, FaServer } from 'react-icons/fa';
import BoardsList from './Boards/BoardsList/BoardsList';
import GalleryBoardName from './GalleryBoardName';
import GallerySettingsPopover from './GallerySettingsPopover';
import GalleryImageGrid from './ImageGrid/GalleryImageGrid';

const selector = createMemoizedSelector([stateSelector], (state) => {
  const { galleryView } = state.gallery;

  return {
    galleryView,
  };
});

const ImageGalleryContent = () => {
  const { t } = useTranslation();
  const resizeObserverRef = useRef<HTMLDivElement>(null);
  const galleryGridRef = useRef<HTMLDivElement>(null);
  const { galleryView } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { isOpen: isBoardListOpen, onToggle: onToggleBoardList } =
    useDisclosure({ defaultIsOpen: true });

  const handleClickImages = useCallback(() => {
    dispatch(galleryViewChanged('images'));
  }, [dispatch]);

  const handleClickAssets = useCallback(() => {
    dispatch(galleryViewChanged('assets'));
  }, [dispatch]);

  return (
    <VStack
      layerStyle="first"
      sx={{
        flexDirection: 'column',
        h: 'full',
        w: 'full',
        borderRadius: 'base',
        p: 2,
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
            variant="unstyled"
            size="sm"
            sx={{ w: 'full' }}
          >
            <TabList>
              <ButtonGroup
                isAttached
                sx={{
                  w: 'full',
                }}
              >
                <Tab
                  as={IAIButton}
                  size="sm"
                  isChecked={galleryView === 'images'}
                  onClick={handleClickImages}
                  sx={{
                    w: 'full',
                  }}
                  leftIcon={<FaImages />}
                  data-testid="images-tab"
                >
                  {t('parameters.images')}
                </Tab>
                <Tab
                  as={IAIButton}
                  size="sm"
                  isChecked={galleryView === 'assets'}
                  onClick={handleClickAssets}
                  sx={{
                    w: 'full',
                  }}
                  leftIcon={<FaServer />}
                  data-testid="assets-tab"
                >
                  {t('gallery.assets')}
                </Tab>
              </ButtonGroup>
            </TabList>
          </Tabs>
        </Flex>
        <GalleryImageGrid />
      </Flex>
    </VStack>
  );
};

export default memo(ImageGalleryContent);

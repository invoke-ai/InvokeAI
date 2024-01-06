import {
  Box,
  Flex,
  Tab,
  TabList,
  Tabs,
  useDisclosure,
  VStack,
} from '@chakra-ui/react';
import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { $galleryHeader } from 'app/store/nanostores/galleryHeader';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvButtonGroup } from 'common/components/InvButtonGroup/InvButtonGroup';
import { galleryViewChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImagesBold } from 'react-icons/pi'
import { RiServerLine } from 'react-icons/ri'

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
  const galleryHeader = useStore($galleryHeader);
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
      flexDirection="column"
      h="full"
      w="full"
      borderRadius="base"
      p={2}
    >
      {galleryHeader}
      <Box w="full">
        <Flex
          ref={resizeObserverRef}
          alignItems="center"
          justifyContent="space-between"
          gap={2}
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
        <Flex alignItems="center" justifyContent="space-between" gap={2}>
          <Tabs
            index={galleryView === 'images' ? 0 : 1}
            variant="unstyled"
            size="sm"
            w="full"
          >
            <TabList>
              <InvButtonGroup w="full">
                <Tab
                  as={InvButton}
                  size="sm"
                  isChecked={galleryView === 'images'}
                  onClick={handleClickImages}
                  w="full"
                  leftIcon={<PiImagesBold size="16px" />}
                  data-testid="images-tab"
                >
                  {t('parameters.images')}
                </Tab>
                <Tab
                  as={InvButton}
                  size="sm"
                  isChecked={galleryView === 'assets'}
                  onClick={handleClickAssets}
                  w="full"
                  leftIcon={<RiServerLine size="16px" />}
                  data-testid="assets-tab"
                >
                  {t('gallery.assets')}
                </Tab>
              </InvButtonGroup>
            </TabList>
          </Tabs>
        </Flex>
        <GalleryImageGrid />
      </Flex>
    </VStack>
  );
};

export default memo(ImageGalleryContent);

import { Box, Button, ButtonGroup, Flex, Tab, TabList, Tabs, useDisclosure } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $galleryHeader } from 'app/store/nanostores/galleryHeader';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { galleryViewChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImagesBold } from 'react-icons/pi';
import { RiServerLine } from 'react-icons/ri';

import BoardsList from './Boards/BoardsList/BoardsList';
import GalleryImageGrid from './ImageGrid/GalleryImageGrid';
import { GalleryPagination } from './ImageGrid/GalleryPagination';
import { GallerySearch } from './ImageGrid/GallerySearch';

const ImageGalleryContent = () => {
  const { t } = useTranslation();
  const galleryView = useAppSelector((s) => s.gallery.galleryView);
  const dispatch = useAppDispatch();
  const galleryHeader = useStore($galleryHeader);
  const { isOpen: isBoardListOpen } = useDisclosure({ defaultIsOpen: true });

  const handleClickImages = useCallback(() => {
    dispatch(galleryViewChanged('images'));
  }, [dispatch]);

  const handleClickAssets = useCallback(() => {
    dispatch(galleryViewChanged('assets'));
  }, [dispatch]);

  return (
    <Flex
      layerStyle="first"
      position="relative"
      flexDirection="column"
      h="full"
      w="full"
      borderRadius="base"
      p={2}
      gap={2}
    >
      {galleryHeader}
      <Box>
        <BoardsList isOpen={isBoardListOpen} />
      </Box>
      <Flex alignItems="center" justifyContent="space-between" gap={2}>
        <Tabs index={galleryView === 'images' ? 0 : 1} variant="unstyled" size="sm" w="full">
          <TabList>
            <ButtonGroup w="full">
              <Tab
                as={Button}
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
                as={Button}
                size="sm"
                isChecked={galleryView === 'assets'}
                onClick={handleClickAssets}
                w="full"
                leftIcon={<RiServerLine size="16px" />}
                data-testid="assets-tab"
              >
                {t('gallery.assets')}
              </Tab>
            </ButtonGroup>
          </TabList>
        </Tabs>
      </Flex>

      <GallerySearch />
      <GalleryImageGrid />
      <GalleryPagination />
    </Flex>
  );
};

export default memo(ImageGalleryContent);

import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, Collapse, Flex, IconButton, Tab, TabList, Tabs, useDisclosure } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $galleryHeader } from 'app/store/nanostores/galleryHeader';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { galleryViewChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImagesBold, PiMagnifyingGlassBold } from 'react-icons/pi';

import BoardsList from './Boards/BoardsList/BoardsList';
import GalleryBoardName from './GalleryBoardName';
import GalleryImageGrid from './ImageGrid/GalleryImageGrid';
import { GalleryPagination } from './ImageGrid/GalleryPagination';
import { GallerySearch } from './ImageGrid/GallerySearch';
import GallerySettingsPopover from './GallerySettingsPopover/GallerySettingsPopover';

const baseStyles: ChakraProps['sx'] = {
  fontWeight: 'semibold',
  fontSize: 'md',
  color: 'base.300',
  borderBottom: '1px solid',
  borderBottomColor: 'invokeBlue.800',
};

const selectedStyles: ChakraProps['sx'] = {
  borderColor: 'invokeBlue.800',
  borderBottomColor: 'base.850',
  color: 'invokeBlue.300',
};

const searchIconStyles: ChakraProps['sx'] = {
  borderBottom: '1px solid',
  borderBottomColor: 'invokeBlue.800',
  maxW: '16',
};

const ImageGalleryContent = () => {
  const { t } = useTranslation();
  const galleryView = useAppSelector((s) => s.gallery.galleryView);
  const dispatch = useAppDispatch();
  const galleryHeader = useStore($galleryHeader);
  const searchDisclosure = useDisclosure({ defaultIsOpen: false });
  const { isOpen: isBoardListOpen, onToggle: onToggleBoardList } = useDisclosure({ defaultIsOpen: true });

  const handleClickImages = useCallback(() => {
    dispatch(galleryViewChanged('images'));
  }, [dispatch]);

  const handleClickAssets = useCallback(() => {
    dispatch(galleryViewChanged('assets'));
  }, [dispatch]);

  return (
    <Flex layerStyle="first" position="relative" flexDirection="column" h="full" w="full" p={2} gap={2}>
      {galleryHeader}
      <Flex alignItems="center" justifyContent="space-between" gap={2}>
          <GalleryBoardName isOpen={isBoardListOpen} onToggle={onToggleBoardList} />
          <GallerySettingsPopover />
        </Flex>
      <BoardsList isOpen={isBoardListOpen}/>
      <Flex alignItems="center" justifyContent="space-between" gap={2}>
        <Tabs isFitted index={galleryView === 'images' ? 0 : 1} variant="enclosed" size="sm" w="full" mb="2">
          <TabList fontSize="sm" borderBottom="none">
            <Tab sx={baseStyles} _selected={selectedStyles} onClick={handleClickImages} data-testid="images-tab">
              <Flex alignItems="center" justifyContent="center" gap="2" w="full">
                <PiImagesBold size="16px" />
                {t('parameters.images')}
              </Flex>
            </Tab>
            <Tab sx={baseStyles} _selected={selectedStyles} onClick={handleClickAssets} data-testid="assets-tab">
              <Flex alignItems="center" justifyContent="center" gap="2" w="full">
                <PiImagesBold size="16px" />
                {t('gallery.assets')}
              </Flex>
            </Tab>
            <Tab sx={searchIconStyles} onClick={searchDisclosure.onToggle}>
            <IconButton
                tooltip={`${t('gallery.displaySearch')}`}
                aria-label={t('gallery.displaySearch')}
                icon={<PiMagnifyingGlassBold />}
                fontSize={16}
                textAlign="center"
                color="base.400"
                variant="unstyled"
                minW="unset"
              />
            </Tab>
          </TabList>
        </Tabs>
      </Flex>
      <Box mb="1">
        <Collapse in={searchDisclosure.isOpen}>
          <GallerySearch />
        </Collapse>
      </Box>
      <GalleryImageGrid />
      <GalleryPagination />
    </Flex>
  );
};

export default memo(ImageGalleryContent);

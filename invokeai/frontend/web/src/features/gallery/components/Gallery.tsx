import type { ChakraProps } from '@invoke-ai/ui-library';
import {
  Box,
  Collapse,
  Flex,
  IconButton,
  Spacer,
  Tab,
  TabList,
  Tabs,
  Text,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGallerySearchTerm } from 'features/gallery/components/ImageGrid/useGallerySearchTerm';
import { galleryViewChanged } from 'features/gallery/store/gallerySlice';
import type { CSSProperties } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMagnifyingGlassBold } from 'react-icons/pi';
import { useBoardName } from 'services/api/hooks/useBoardName';

import GalleryImageGrid from './ImageGrid/GalleryImageGrid';
import { GalleryPagination } from './ImageGrid/GalleryPagination';
import { GallerySearch } from './ImageGrid/GallerySearch';

const BASE_STYLES: ChakraProps['sx'] = {
  fontWeight: 'semibold',
  fontSize: 'sm',
  color: 'base.300',
};

const SELECTED_STYLES: ChakraProps['sx'] = {
  borderColor: 'base.800',
  borderBottomColor: 'base.900',
  color: 'invokeBlue.300',
};

const COLLAPSE_STYLES: CSSProperties = { flexShrink: 0, minHeight: 0 };

export const Gallery = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const galleryView = useAppSelector((s) => s.gallery.galleryView);
  const initialSearchTerm = useAppSelector((s) => s.gallery.searchTerm);
  const searchDisclosure = useDisclosure({ defaultIsOpen: initialSearchTerm.length > 0 });
  const [searchTerm, onChangeSearchTerm, onResetSearchTerm] = useGallerySearchTerm();

  const handleClickImages = useCallback(() => {
    dispatch(galleryViewChanged('images'));
  }, [dispatch]);

  const handleClickAssets = useCallback(() => {
    dispatch(galleryViewChanged('assets'));
  }, [dispatch]);

  const handleClickSearch = useCallback(() => {
    searchDisclosure.onToggle();
    onResetSearchTerm();
  }, [onResetSearchTerm, searchDisclosure]);

  const selectedBoardId = useAppSelector((s) => s.gallery.selectedBoardId);
  const boardName = useBoardName(selectedBoardId);

  return (
    <Flex flexDirection="column" alignItems="center" justifyContent="space-between" h="full" w="full" pt={1}>
      <Tabs index={galleryView === 'images' ? 0 : 1} variant="enclosed" display="flex" flexDir="column" w="full">
        <TabList gap={2} fontSize="sm" borderColor="base.800" alignItems="center" w="full">
          <Text fontSize="sm" fontWeight="semibold" noOfLines={1} px="2" wordBreak="break-all">
            {boardName}
          </Text>
          <Spacer />
          <Tab sx={BASE_STYLES} _selected={SELECTED_STYLES} onClick={handleClickImages} data-testid="images-tab">
            {t('parameters.images')}
          </Tab>
          <Tab sx={BASE_STYLES} _selected={SELECTED_STYLES} onClick={handleClickAssets} data-testid="assets-tab">
            {t('gallery.assets')}
          </Tab>
          <IconButton
            onClick={handleClickSearch}
            tooltip={searchDisclosure.isOpen ? `${t('gallery.exitSearch')}` : `${t('gallery.displaySearch')}`}
            aria-label={t('gallery.displaySearch')}
            icon={<PiMagnifyingGlassBold />}
            colorScheme={searchDisclosure.isOpen ? 'invokeBlue' : 'base'}
            variant="link"
          />
        </TabList>
      </Tabs>

      <Box w="full">
        <Collapse in={searchDisclosure.isOpen} style={COLLAPSE_STYLES}>
          <Box w="full" pt={2}>
            <GallerySearch
              searchTerm={searchTerm}
              onChangeSearchTerm={onChangeSearchTerm}
              onResetSearchTerm={onResetSearchTerm}
            />
          </Box>
        </Collapse>
      </Box>
      <GalleryImageGrid />
      <GalleryPagination />
    </Flex>
  );
};

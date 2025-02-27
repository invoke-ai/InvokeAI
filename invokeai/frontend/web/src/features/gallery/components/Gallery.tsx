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
  Tooltip,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGallerySearchTerm } from 'features/gallery/components/ImageGrid/useGallerySearchTerm';
import { selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { galleryViewChanged, selectGallerySlice } from 'features/gallery/store/gallerySlice';
import type { CSSProperties } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMagnifyingGlassBold } from 'react-icons/pi';
import { useBoardName } from 'services/api/hooks/useBoardName';

import GallerySettingsPopover from './GallerySettingsPopover/GallerySettingsPopover';
import { GalleryUploadButton } from './GalleryUploadButton';
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

const COLLAPSE_STYLES: CSSProperties = { flexShrink: 0, minHeight: 0, width: '100%' };

const selectGalleryView = createSelector(selectGallerySlice, (gallery) => gallery.galleryView);
const selectSearchTerm = createSelector(selectGallerySlice, (gallery) => gallery.searchTerm);

export const Gallery = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const galleryView = useAppSelector(selectGalleryView);
  const initialSearchTerm = useAppSelector(selectSearchTerm);
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

  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const boardName = useBoardName(selectedBoardId);

  return (
    <Flex flexDirection="column" alignItems="center" justifyContent="space-between" h="full" w="full" pt={1} minH={0}>
      <Tabs index={galleryView === 'images' ? 0 : 1} variant="enclosed" display="flex" flexDir="column" w="full">
        <TabList gap={2} fontSize="sm" borderColor="base.800" alignItems="center" w="full">
          <Text fontSize="sm" fontWeight="semibold" noOfLines={1} px="2" wordBreak="break-all">
            {boardName}
          </Text>
          <Spacer />
          <Tooltip label={t('gallery.imagesTab')}>
            <Tab sx={BASE_STYLES} _selected={SELECTED_STYLES} onClick={handleClickImages} data-testid="images-tab">
              {t('parameters.images')}
            </Tab>
          </Tooltip>
          <Tooltip label={t('gallery.assetsTab')}>
            <Tab sx={BASE_STYLES} _selected={SELECTED_STYLES} onClick={handleClickAssets} data-testid="assets-tab">
              {t('gallery.assets')}
            </Tab>
          </Tooltip>
          <Flex h="full" justifyContent="flex-end">
            <GalleryUploadButton />
            <GallerySettingsPopover />
            <IconButton
              size="sm"
              variant="link"
              alignSelf="stretch"
              onClick={handleClickSearch}
              tooltip={searchDisclosure.isOpen ? `${t('gallery.exitSearch')}` : `${t('gallery.displaySearch')}`}
              aria-label={t('gallery.displaySearch')}
              icon={<PiMagnifyingGlassBold />}
            />
          </Flex>
        </TabList>
      </Tabs>

      <Collapse in={searchDisclosure.isOpen} style={COLLAPSE_STYLES}>
        <Box w="full" pt={2}>
          <GallerySearch
            searchTerm={searchTerm}
            onChangeSearchTerm={onChangeSearchTerm}
            onResetSearchTerm={onResetSearchTerm}
          />
        </Box>
      </Collapse>
      <GalleryImageGrid />
      <GalleryPagination />
    </Flex>
  );
};

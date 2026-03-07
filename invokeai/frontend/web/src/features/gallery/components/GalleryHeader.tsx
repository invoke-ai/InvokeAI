import { Button, ButtonGroup, Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { GalleryUploadButton } from 'features/gallery/components/GalleryUploadButton';
import { GallerySearch } from 'features/gallery/components/ImageGrid/GallerySearch';
import { useGallerySearchTerm } from 'features/gallery/components/ImageGrid/useGallerySearchTerm';
import { useGalleryImageNames } from 'features/gallery/components/use-gallery-image-names';
import { selectSelectedBoardId, selectShowAspectRatioThumbnails } from 'features/gallery/store/gallerySelectors';
import {
  galleryViewChanged,
  selectGallerySlice,
  showAspectRatioThumbnailsChanged,
} from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArchiveBold, PiGridNineBold, PiImageSquareBold, PiSquaresFourBold } from 'react-icons/pi';
import { useBoardName } from 'services/api/hooks/useBoardName';

import { GallerySettingsPopover } from './GallerySettingsPopover/GallerySettingsPopover';

const selectGalleryView = createSelector(selectGallerySlice, (gallery) => gallery.galleryView);
const selectSearchTerm = createSelector(selectGallerySlice, (gallery) => gallery.searchTerm);

export const GalleryHeader = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const boardName = useBoardName(selectedBoardId);
  const showAspectRatio = useAppSelector(selectShowAspectRatioThumbnails);
  const galleryView = useAppSelector(selectGalleryView);
  const { imageNames } = useGalleryImageNames();
  const hasImages = imageNames.length > 0;

  const initialSearchTerm = useAppSelector(selectSearchTerm);
  const [searchTerm, onChangeSearchTerm, onResetSearchTerm] = useGallerySearchTerm();

  const handleToggleAspectRatio = useCallback(() => {
    dispatch(showAspectRatioThumbnailsChanged(!showAspectRatio));
  }, [dispatch, showAspectRatio]);

  const handleClickImages = useCallback(() => {
    dispatch(galleryViewChanged('images'));
  }, [dispatch]);

  const handleClickAssets = useCallback(() => {
    dispatch(galleryViewChanged('assets'));
  }, [dispatch]);

  return (
    <Flex gap={2} borderBottomWidth={1} borderColor="base.750" alignItems="center" h={12} px={2} w="full">
      <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.400">
        {boardName || t('gallery.allImages')}
      </Text>
      <Spacer />
      <ButtonGroup size="sm" variant="outline">
        <Button
          leftIcon={<PiImageSquareBold />}
          tooltip={t('gallery.imagesTab')}
          onClick={handleClickImages}
          data-testid="images-tab"
          colorScheme={galleryView === 'images' ? 'invokeBlue' : undefined}
        >
          {t('parameters.images')}
        </Button>
        <Button
          leftIcon={<PiArchiveBold />}
          tooltip={t('gallery.assetsTab')}
          onClick={handleClickAssets}
          data-testid="assets-tab"
          colorScheme={galleryView === 'assets' ? 'invokeBlue' : undefined}
        >
          {t('gallery.assets')}
        </Button>
      </ButtonGroup>
      <Flex maxW={64} flex={1}>
        <GallerySearch
          searchTerm={searchTerm || initialSearchTerm}
          onChangeSearchTerm={onChangeSearchTerm}
          onResetSearchTerm={onResetSearchTerm}
          isDisabled={!hasImages}
        />
      </Flex>
      <Flex h="full">
        <IconButton
          size="sm"
          variant="link"
          alignSelf="stretch"
          onClick={handleToggleAspectRatio}
          tooltip={showAspectRatio ? t('gallery.showSquareThumbnails') : t('gallery.showAspectRatioThumbnails')}
          aria-label={t('gallery.toggleAspectRatio')}
          icon={showAspectRatio ? <PiGridNineBold /> : <PiSquaresFourBold />}
          colorScheme={showAspectRatio ? 'invokeBlue' : 'base'}
        />
        <GalleryUploadButton />
        <GallerySettingsPopover />
      </Flex>
    </Flex>
  );
});

GalleryHeader.displayName = 'GalleryHeader';

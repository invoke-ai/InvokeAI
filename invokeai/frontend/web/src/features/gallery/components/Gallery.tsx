import { Box, Button, ButtonGroup, Collapse, Divider, Flex, IconButton, Spacer } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useDisclosure } from 'common/hooks/useBoolean';
import { useGallerySearchTerm } from 'features/gallery/components/ImageGrid/useGallerySearchTerm';
import { selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { galleryViewChanged, selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { useAutoLayoutContext } from 'features/ui/layouts/auto-layout-context';
import {
  GALLERY_PANEL_DEFAULT_HEIGHT_PX,
  GALLERY_PANEL_ID,
  GALLERY_PANEL_MIN_HEIGHT_PX,
} from 'features/ui/layouts/shared';
import { useCollapsibleGridviewPanel } from 'features/ui/layouts/use-collapsible-gridview-panel';
import type { CSSProperties } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiCaretUpBold, PiMagnifyingGlassBold } from 'react-icons/pi';
import { useBoardName } from 'services/api/hooks/useBoardName';

import { GallerySettingsPopover } from './GallerySettingsPopover/GallerySettingsPopover';
import { GalleryUploadButton } from './GalleryUploadButton';
import { GallerySearch } from './ImageGrid/GallerySearch';
import { NewGallery } from './NewGallery';

const COLLAPSE_STYLES: CSSProperties = { flexShrink: 0, minHeight: 0, width: '100%' };

const selectGalleryView = createSelector(selectGallerySlice, (gallery) => gallery.galleryView);
const selectSearchTerm = createSelector(selectGallerySlice, (gallery) => gallery.searchTerm);

export const GalleryPanel = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { _$rightPanelApi } = useAutoLayoutContext();
  const gridviewPanelApi = useStore(_$rightPanelApi);
  const collapsibleApi = useCollapsibleGridviewPanel(
    gridviewPanelApi,
    GALLERY_PANEL_ID,
    'vertical',
    GALLERY_PANEL_DEFAULT_HEIGHT_PX,
    GALLERY_PANEL_MIN_HEIGHT_PX
  );
  const isCollapsed = useStore(collapsibleApi.$isCollapsed);

  const galleryView = useAppSelector(selectGalleryView);
  const initialSearchTerm = useAppSelector(selectSearchTerm);
  const searchDisclosure = useDisclosure(!!initialSearchTerm);
  const [searchTerm, onChangeSearchTerm, onResetSearchTerm] = useGallerySearchTerm();
  const handleClickImages = useCallback(() => {
    dispatch(galleryViewChanged('images'));
  }, [dispatch]);

  const handleClickAssets = useCallback(() => {
    dispatch(galleryViewChanged('assets'));
  }, [dispatch]);

  const handleClickSearch = useCallback(() => {
    onResetSearchTerm();
    if (!searchDisclosure.isOpen && collapsibleApi.$isCollapsed.get()) {
      collapsibleApi.expand();
    }
    searchDisclosure.toggle();
  }, [collapsibleApi, onResetSearchTerm, searchDisclosure]);

  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const boardName = useBoardName(selectedBoardId);

  return (
    <Flex flexDirection="column" alignItems="center" justifyContent="space-between" h="full" w="full" minH={0}>
      <Flex gap={2} fontSize="sm" alignItems="center" w="full">
        <Button
          size="sm"
          variant="ghost"
          onClick={collapsibleApi.toggle}
          leftIcon={isCollapsed ? <PiCaretDownBold /> : <PiCaretUpBold />}
        >
          {boardName}
        </Button>
        <Spacer />
        <ButtonGroup size="sm" variant="outline">
          <Button
            tooltip={t('gallery.imagesTab')}
            onClick={handleClickImages}
            data-testid="images-tab"
            colorScheme={galleryView === 'images' ? 'invokeBlue' : undefined}
          >
            {t('parameters.images')}
          </Button>
          <Button
            tooltip={t('gallery.assetsTab')}
            onClick={handleClickAssets}
            data-testid="assets-tab"
            colorScheme={galleryView === 'assets' ? 'invokeBlue' : undefined}
          >
            {t('gallery.assets')}
          </Button>
        </ButtonGroup>
        <Flex flexGrow={1} flexBasis={0} justifyContent="flex-end">
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
      </Flex>
      <Collapse in={searchDisclosure.isOpen} style={COLLAPSE_STYLES}>
        <Box w="full" pt={2}>
          <GallerySearch
            searchTerm={searchTerm}
            onChangeSearchTerm={onChangeSearchTerm}
            onResetSearchTerm={onResetSearchTerm}
          />
        </Box>
      </Collapse>
      <Divider pt={2} />
      <Flex w="full" h="full" pt={2}>
        <NewGallery />
      </Flex>
    </Flex>
  );
});
GalleryPanel.displayName = 'GalleryPanel';

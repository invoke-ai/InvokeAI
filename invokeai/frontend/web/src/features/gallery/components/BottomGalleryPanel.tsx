import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Button, ButtonGroup, Collapse, Flex, IconButton, Spacer } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { useDisclosure } from 'common/hooks/useBoolean';
import AddBoardButton from 'features/gallery/components/Boards/BoardsList/AddBoardButton';
import { BoardsList } from 'features/gallery/components/Boards/BoardsList/BoardsList';
import { BoardsSearch } from 'features/gallery/components/Boards/BoardsList/BoardsSearch';
import { GalleryImageGrid } from 'features/gallery/components/GalleryImageGrid';
import { GalleryImageGridPaged } from 'features/gallery/components/GalleryImageGridPaged';
import { GallerySettingsPopover } from 'features/gallery/components/GallerySettingsPopover/GallerySettingsPopover';
import { GalleryUploadButton } from 'features/gallery/components/GalleryUploadButton';
import { GallerySearch } from 'features/gallery/components/ImageGrid/GallerySearch';
import { useGallerySearchTerm } from 'features/gallery/components/ImageGrid/useGallerySearchTerm';
import { selectSelectedBoardId, selectShowAspectRatioThumbnails } from 'features/gallery/store/gallerySelectors';
import {
  galleryViewChanged,
  selectGallerySlice,
  showAspectRatioThumbnailsChanged,
} from 'features/gallery/store/gallerySlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { BOARDS_SIDEBAR_DEFAULT_WIDTH_PX, BOARDS_SIDEBAR_MIN_WIDTH_PX } from 'features/ui/layouts/shared';
import { selectShouldUsePagedGalleryView } from 'features/ui/store/uiSelectors';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties, MouseEvent as ReactMouseEvent } from 'react';
import { memo, useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiCaretDownBold,
  PiCaretUpBold,
  PiGridNineBold,
  PiMagnifyingGlassBold,
  PiSidebarSimpleBold,
  PiSquaresFourBold,
} from 'react-icons/pi';
import { useBoardName } from 'services/api/hooks/useBoardName';

const COLLAPSE_STYLES: CSSProperties = { flexShrink: 0, minHeight: 0, width: '100%' };

const HEADER_STYLES_SX: SystemStyleObject = {
  gap: 2,
  p: 2,
  alignItems: 'center',
  w: 'full',
  flexShrink: 0,
  borderBottomWidth: 1,
  borderColor: 'base.700',
  bg: 'base.850',
}

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

const selectGalleryView = createSelector(selectGallerySlice, (gallery) => gallery.galleryView);
const selectSearchTerm = createSelector(selectGallerySlice, (gallery) => gallery.searchTerm);

/**
 * Ensures the bottom gallery panel is expanded. If it's currently collapsed, expands it.
 * This is called by action handlers in the header bar so that clicking any action
 * while collapsed also opens the gallery.
 */
const ensureExpanded = () => {
  if (navigationApi.isBottomPanelCollapsed()) {
    navigationApi.expandBottomPanel();
  }
};

/**
 * The bottom gallery panel that contains the boards sidebar and image grid.
 * This component is placed at the bottom of the layout spanning the full width.
 */
export const BottomGalleryPanel = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const galleryView = useAppSelector(selectGalleryView);
  const initialSearchTerm = useAppSelector(selectSearchTerm);
  const shouldUsePagedGalleryView = useAppSelector(selectShouldUsePagedGalleryView);
  const showAspectRatio = useAppSelector(selectShowAspectRatioThumbnails);
  const searchDisclosure = useDisclosure(!!initialSearchTerm);
  const [searchTerm, onChangeSearchTerm, onResetSearchTerm] = useGallerySearchTerm();
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const boardName = useBoardName(selectedBoardId);
  const [isBoardsSidebarCollapsed, setIsBoardsSidebarCollapsed] = useState(false);
  const [boardsSidebarWidth, setBoardsSidebarWidth] = useState(BOARDS_SIDEBAR_DEFAULT_WIDTH_PX);
  const isResizingRef = useRef(false);
  const [isGalleryCollapsed, setIsGalleryCollapsed] = useState(false);

  const handleClickImages = useCallback(() => {
    ensureExpanded();
    setIsGalleryCollapsed(false);
    dispatch(galleryViewChanged('images'));
  }, [dispatch]);

  const handleClickAssets = useCallback(() => {
    ensureExpanded();
    setIsGalleryCollapsed(false);
    dispatch(galleryViewChanged('assets'));
  }, [dispatch]);

  const handleClickSearch = useCallback(() => {
    ensureExpanded();
    setIsGalleryCollapsed(false);
    onResetSearchTerm();
    searchDisclosure.toggle();
  }, [onResetSearchTerm, searchDisclosure]);

  const handleToggleGallery = useCallback(() => {
    setIsGalleryCollapsed((prev) => !prev);
    navigationApi.toggleBottomPanel();
  }, []);

  const handleToggleBoardsSidebar = useCallback(() => {
    ensureExpanded();
    setIsGalleryCollapsed(false);
    setIsBoardsSidebarCollapsed((prev) => !prev);
  }, []);

  const handleToggleAspectRatio = useCallback(() => {
    ensureExpanded();
    setIsGalleryCollapsed(false);
    dispatch(showAspectRatioThumbnailsChanged(!showAspectRatio));
  }, [dispatch, showAspectRatio]);

  // Resize handler for the boards sidebar
  const handleResizeStart = useCallback(
    (e: ReactMouseEvent) => {
      if (isBoardsSidebarCollapsed) {
        return;
      }
      e.preventDefault();
      isResizingRef.current = true;
      const startX = e.clientX;
      const startWidth = boardsSidebarWidth;

      const onMouseMove = (moveEvent: MouseEvent) => {
        if (!isResizingRef.current) {
          return;
        }
        const delta = moveEvent.clientX - startX;
        const newWidth = Math.max(BOARDS_SIDEBAR_MIN_WIDTH_PX, startWidth + delta);
        setBoardsSidebarWidth(newWidth);
      };

      const onMouseUp = () => {
        isResizingRef.current = false;
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };

      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    },
    [boardsSidebarWidth, isBoardsSidebarCollapsed]
  );

  return (
    <Flex flexDirection="column" h="full" w="full" minH={0}>
      {/* Header bar - always visible, even when collapsed */}
      <Flex sx={HEADER_STYLES_SX}>
        <Flex gap={2} alignItems="center">
          <Button
            size="sm"
            variant="ghost"
            onClick={handleToggleGallery}
            leftIcon={isGalleryCollapsed ? <PiCaretUpBold /> : <PiCaretDownBold />}
            // noOfLines={1}
            flexShrink={0}
            tooltip={isGalleryCollapsed ? t('gallery.expandGallery') : t('gallery.collapseGallery')}
          >
            {t('common.board')}: {boardName}
          </Button>
          <IconButton
            size="sm"
            variant="link"
            alignSelf="stretch"
            onClick={handleToggleBoardsSidebar}
            tooltip={isBoardsSidebarCollapsed ? t('gallery.showBoardsSidebar') : t('gallery.hideBoardsSidebar')}
            aria-label={t('gallery.toggleBoardsSidebar')}
            icon={<PiSidebarSimpleBold />}
            colorScheme={isBoardsSidebarCollapsed ? 'base' : 'invokeBlue'}
          />
        </Flex>
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
        <Flex gap={0} alignItems="center">
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
          <IconButton
            size="sm"
            variant="link"
            alignSelf="stretch"
            onClick={handleClickSearch}
            tooltip={searchDisclosure.isOpen ? `${t('gallery.exitSearch')}` : `${t('gallery.displaySearch')}`}
            aria-label={t('gallery.displaySearch')}
            icon={<PiMagnifyingGlassBold />}
            colorScheme={searchDisclosure.isOpen ? 'invokeBlue' : 'base'}
          />
        </Flex>
      </Flex>

      {/* Main content area - hidden when collapsed */}
      <Flex flex={1} minH={0} w="full">
        {/* Boards Sidebar */}
        {!isBoardsSidebarCollapsed && (
          <>
            <Flex
              flexDirection="column"
              w={`${boardsSidebarWidth}px`}
              minW={`${BOARDS_SIDEBAR_MIN_WIDTH_PX}px`}
              h="full"
              flexShrink={0}
              borderRightWidth={1}
              borderColor="base.700"
              gap={1}
            >
              {/* Boards search at top */}
              <Flex px={2} pt={2} flexShrink={0}>
                <BoardsSearch />
              </Flex>

              {/* Board list - scrollable middle */}
              <Flex flex={1} minH={0} position="relative">
                <Box position="absolute" top={0} right={0} bottom={0} left={0} px={2}>
                  <OverlayScrollbarsComponent style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
                    <BoardsList />
                  </OverlayScrollbarsComponent>
                </Box>
              </Flex>

              {/* Sticky Add Board footer */}
              <Flex
                p={2}
                borderTopWidth={1}
                borderColor="base.700"
                alignItems="center"
                justifyContent="center"
                flexShrink={0}
              >
                <AddBoardButton />
              </Flex>
            </Flex>

            {/* Resize handle */}
            <Flex
              w="4px"
              cursor="col-resize"
              bg="base.700"
              _hover={{ bg: 'base.500' }}
              transition="background 0.15s"
              onMouseDown={handleResizeStart}
              flexShrink={0}
            />
          </>
        )}

        {/* Image grid area */}
        <Flex flexDirection="column" flex={1} minW={0} h="full">
          {/* Search bar (collapsible) */}
          <Collapse in={searchDisclosure.isOpen} style={COLLAPSE_STYLES}>
            <Box w="full" px={2} pt={2}>
              <GallerySearch
                searchTerm={searchTerm}
                onChangeSearchTerm={onChangeSearchTerm}
                onResetSearchTerm={onResetSearchTerm}
              />
            </Box>
          </Collapse>

          {/* Image grid */}
          <Flex w="full" flex={1} minH={0} p={2}>
            {shouldUsePagedGalleryView ? <GalleryImageGridPaged /> : <GalleryImageGrid />}
          </Flex>
        </Flex>
      </Flex>
    </Flex>
  );
});

BottomGalleryPanel.displayName = 'BottomGalleryPanel';

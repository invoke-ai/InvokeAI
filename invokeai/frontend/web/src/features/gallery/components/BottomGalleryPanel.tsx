import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Button, Flex, IconButton, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import AddBoardButton from 'features/gallery/components/Boards/BoardsList/AddBoardButton';
import { BoardsList } from 'features/gallery/components/Boards/BoardsList/BoardsList';
import { GalleryImageGrid } from 'features/gallery/components/GalleryImageGrid';
import { GalleryImageGridPaged } from 'features/gallery/components/GalleryImageGridPaged';
import { selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
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
  PiMagnifyingGlassBold,
  PiSidebarSimpleBold,
} from 'react-icons/pi';
import { useBoardName } from 'services/api/hooks/useBoardName';

import { GalleryHeader } from './GalleryHeader';

const HEADER_STYLES_SX: SystemStyleObject = {
  gap: 2,
  p: 2,
  alignItems: 'center',
  w: 'full',
  flexShrink: 0,
  borderBottomWidth: 1,
  borderColor: 'base.700',
  bg: 'base.850',
  h: 12,
};

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

const BOARDS_SIDEBAR_STYLES_SX: SystemStyleObject = {
  h: 'full',
  flexShrink: 0,
  borderEndWidth: 1,
  borderColor: 'base.700',
  gap: 1,
  position: 'relative',
};

const BOARDS_SIDEBAR_EXPANDED_STYLES_SX: SystemStyleObject = {
  ...BOARDS_SIDEBAR_STYLES_SX,
  w: `${BOARDS_SIDEBAR_DEFAULT_WIDTH_PX}px`,
  minW: `${BOARDS_SIDEBAR_MIN_WIDTH_PX}px`,
};

const BOARDS_SIDEBAR_COLLAPSED_STYLES_SX: SystemStyleObject = {
  ...BOARDS_SIDEBAR_STYLES_SX,
  w: 12,
  minW: 12,
};

/**
 * The bottom gallery panel that contains the boards sidebar and image grid.
 * This component is placed at the bottom of the layout spanning the full width.
 */
export const BottomGalleryPanel = memo(() => {
  const { t } = useTranslation();
  const shouldUsePagedGalleryView = useAppSelector(selectShouldUsePagedGalleryView);
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const boardName = useBoardName(selectedBoardId);
  const [isBoardsSidebarCollapsed, setIsBoardsSidebarCollapsed] = useState(false);
  const [boardsSidebarWidth, setBoardsSidebarWidth] = useState(BOARDS_SIDEBAR_DEFAULT_WIDTH_PX);
  const isResizingRef = useRef(false);
  const [isGalleryCollapsed, setIsGalleryCollapsed] = useState(false);

  const handleToggleGallery = useCallback(() => {
    setIsGalleryCollapsed((prev) => !prev);
    navigationApi.toggleBottomPanel();
  }, []);

  const handleToggleBoardsSidebar = useCallback(() => {
    setIsBoardsSidebarCollapsed((prev) => !prev);
  }, []);

  const onCollapsedBoardSearch = useCallback(() => {
    if (isBoardsSidebarCollapsed) {
      setIsBoardsSidebarCollapsed(false);
    }
    // TODO: Highlight the search input in the sidebar after expanding
  }, [isBoardsSidebarCollapsed]);

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
      </Flex>

      {/* Main content area - hidden when collapsed */}
      <Flex flex={1} minH={0} w="full">
        {/* Boards Sidebar */}

        {/*{!isBoardsSidebarCollapsed && (
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
              <Flex flex={1} minH={0} position="relative">
                <Box position="absolute" top={0} right={0} bottom={0} left={0} px={2}>
                  <OverlayScrollbarsComponent style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
                    <BoardsList />
                  </OverlayScrollbarsComponent>
                </Box>
              </Flex>


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
        )}*/}

        <Flex
          flexDirection="column"
          sx={isBoardsSidebarCollapsed ? BOARDS_SIDEBAR_COLLAPSED_STYLES_SX : BOARDS_SIDEBAR_EXPANDED_STYLES_SX}
        >
          {isBoardsSidebarCollapsed ? (
            <>
              <IconButton
                size="sm"
                variant="ghost"
                icon={<PiSidebarSimpleBold />}
                onClick={handleToggleBoardsSidebar}
                tooltip={t('gallery.showBoardsSidebar')}
                aria-label={t('gallery.toggleBoardsSidebar')}
              />
              <IconButton
                size="sm"
                variant="ghost"
                icon={<PiMagnifyingGlassBold />}
                onClick={onCollapsedBoardSearch}
                tooltip={t('gallery.searchBoards')}
                aria-label={t('gallery.searchBoards')}
              />
            </>
          ) : (
            <p>something</p>
          )}
        </Flex>

        {!isBoardsSidebarCollapsed && (
          <Flex
            w="4px"
            cursor="col-resize"
            bg="base.700"
            _hover={{ bg: 'base.500' }}
            transition="background 0.15s"
            onMouseDown={handleResizeStart}
            flexShrink={0}
          />
        )}

        {/* Image grid area */}
        <Flex flexDirection="column" flex={1} minW={800} h="full">
          <GalleryHeader />

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

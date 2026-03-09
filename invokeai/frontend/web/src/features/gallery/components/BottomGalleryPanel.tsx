import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Button, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import type { GridviewPanel } from 'dockview';
import { AddBoardButton, AddBoardIconButton } from 'features/gallery/components/Boards/BoardsList/AddBoardButton';
import { BoardsList } from 'features/gallery/components/Boards/BoardsList/BoardsList';
import { GalleryImageGrid } from 'features/gallery/components/GalleryImageGrid';
import { GalleryImageGridPaged } from 'features/gallery/components/GalleryImageGridPaged';
import { selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { useAutoLayoutContext } from 'features/ui/layouts/auto-layout-context';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import {
  BOARDS_SIDEBAR_DEFAULT_WIDTH_PX,
  BOARDS_SIDEBAR_MAX_WIDTH_PX,
  BOARDS_SIDEBAR_MIN_WIDTH_PX,
  BOTTOM_GALLERY_MIN_HEIGHT_PX,
  BOTTOM_GALLERY_PANEL_ID,
} from 'features/ui/layouts/shared';
import { selectShouldUsePagedGalleryView } from 'features/ui/store/uiSelectors';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties, MouseEvent as ReactMouseEvent } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiCaretUpBold } from 'react-icons/pi';
import { useBoardName } from 'services/api/hooks/useBoardName';

import { GalleryHeader } from './GalleryHeader';

const HEADER_STYLES_SX: SystemStyleObject = {
  gap: 2,
  p: 2,
  alignItems: 'center',
  w: 'full',
  flexShrink: 0,
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
  position: 'relative',
};

const BOARDS_SIDEBAR_EXPANDED_STYLES_SX: SystemStyleObject = {
  ...BOARDS_SIDEBAR_STYLES_SX,
  minW: `${BOARDS_SIDEBAR_MIN_WIDTH_PX}px`,
};

const COLLAPSED_SIDEBAR_WIDTH = 16;

const BOARDS_SIDEBAR_COLLAPSED_STYLES_SX: SystemStyleObject = {
  ...BOARDS_SIDEBAR_STYLES_SX,
  w: COLLAPSED_SIDEBAR_WIDTH,
  minW: COLLAPSED_SIDEBAR_WIDTH,
};

const BOARDS_SIDEBAR_RESIZE_HANDLE_STYLES_SX: SystemStyleObject = {
  position: 'absolute',
  insetY: 0,
  right: '-2px',
  w: '4px',
  cursor: 'col-resize',
  _hover: { bg: 'base.600' },
  transition: 'background 0.15s',
};

/**
 * Hook that derives the gallery panel's collapsed state from dockview's panel dimensions.
 * Subscribes to the panel's onDidDimensionsChange event so the UI stays in sync
 * regardless of how the panel is collapsed/expanded (toggle button, hotkey, or external API call).
 */
const useIsGalleryPanelCollapsed = (): boolean => {
  const { tab } = useAutoLayoutContext();
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    const panel = navigationApi.getPanel(tab, BOTTOM_GALLERY_PANEL_ID) as GridviewPanel | undefined;
    if (!panel) {
      return;
    }

    // Sync initial state
    setIsCollapsed(panel.height <= BOTTOM_GALLERY_MIN_HEIGHT_PX);

    // Subscribe to dimension changes
    const { dispose } = panel.api.onDidDimensionsChange((event) => {
      setIsCollapsed(event.height <= BOTTOM_GALLERY_MIN_HEIGHT_PX);
    });

    return dispose;
  }, [tab]);

  return isCollapsed;
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
  const isGalleryPanelCollapsed = useIsGalleryPanelCollapsed();

  const handleToggleGalleryPanel = useCallback(() => {
    navigationApi.toggleBottomPanel();
  }, []);

  const handleToggleBoardsSidebar = useCallback(() => {
    setIsBoardsSidebarCollapsed((prev) => !prev);
  }, []);

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
        const newWidth = Math.min(
          BOARDS_SIDEBAR_MAX_WIDTH_PX,
          Math.max(BOARDS_SIDEBAR_MIN_WIDTH_PX, startWidth + delta)
        );
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
      {/* Top header bar */}
      <Flex sx={HEADER_STYLES_SX} borderBottomWidth={isGalleryPanelCollapsed ? 0 : 1}>
        <Button
          size="sm"
          variant="ghost"
          onClick={handleToggleGalleryPanel}
          leftIcon={isGalleryPanelCollapsed ? <PiCaretUpBold /> : <PiCaretDownBold />}
          // noOfLines={1} -> this results in weird vertical alignment, ideally we re-enable this and fix
          flexShrink={0}
          tooltip={isGalleryPanelCollapsed ? t('gallery.expandGallery') : t('gallery.collapseGallery')}
        >
          {t('common.board')}: {boardName}
        </Button>
      </Flex>

      {/* Main content area - hidden when collapsed */}
      <Flex flex={1} minH={0} w="full" position="relative">
        {/* Boards Sidebar */}
        <Flex
          flexDirection="column"
          sx={isBoardsSidebarCollapsed ? BOARDS_SIDEBAR_COLLAPSED_STYLES_SX : BOARDS_SIDEBAR_EXPANDED_STYLES_SX}
          w={isBoardsSidebarCollapsed ? COLLAPSED_SIDEBAR_WIDTH : `${boardsSidebarWidth}px`}
        >
          <Flex flex={1} minH={0} position="relative">
            <Box position="absolute" top={0} right={0} bottom={0} left={0} px={2}>
              <OverlayScrollbarsComponent style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
                <BoardsList
                  onCollapseBoards={handleToggleBoardsSidebar}
                  onExpandBoards={handleToggleBoardsSidebar}
                  isCollapsed={isBoardsSidebarCollapsed}
                />
              </OverlayScrollbarsComponent>
            </Box>
          </Flex>

          <Flex p={1} borderTopWidth={1} borderColor="base.700" justifyContent="center" flexShrink={0} h={10}>
            {isBoardsSidebarCollapsed ? <AddBoardIconButton /> : <AddBoardButton />}
          </Flex>

          {!isBoardsSidebarCollapsed && (
            <Flex sx={BOARDS_SIDEBAR_RESIZE_HANDLE_STYLES_SX} onMouseDown={handleResizeStart} />
          )}
        </Flex>

        {/* Image grid area */}
        <Flex flexDirection="column" flex={1} minW={800} h="full">
          {/* Gallery controls, settings + search */}
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

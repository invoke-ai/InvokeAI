import { Box, Flex, HStack, Spacer, Stack } from '@chakra-ui/react';
import { GALLERY_BOARD_PANEL_MAX_WIDTH_PX, GALLERY_BOARD_PANEL_MIN_WIDTH_PX } from '@features/gallery/core/settings';
import { ResizeHandle } from '@platform/ui/ResizeHandle';
import { segmentTabsPanelId, segmentTabsTabId } from '@platform/ui/SegmentTabs';
import { useCallback, useId, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { GalleryBoardsPanel } from './GalleryBoardsPanel';
import { GalleryImageGrid } from './GalleryImageGrid';
import { GalleryItemSearch } from './GalleryItemSearch';
import { GalleryItemSortMenu } from './GalleryItemSortMenu';
import { GallerySelectionBar } from './GallerySelectionBar';
import { GalleryStarredFilterToggle } from './GalleryStarredFilterToggle';
import { GalleryUploadButton } from './GalleryUploadButton';
import { GalleryViewTabs } from './GalleryViewTabs';
import { useGalleryWidget } from './GalleryWidgetContext';

const CHROME_INSET_PADDING_TOP = 'calc(var(--chakra-spacing-2) + var(--wb-center-chrome-inset, 0px))';

export const GalleryWideLayout = () => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const { boardPanelCollapsed, boardPanelWidthPx } = gallery.settings;
  const viewTabsIdBase = useId();
  const boardPanelRef = useRef<HTMLDivElement>(null);

  const handleCommitWidth = useCallback(
    (boardPanelWidthPx: number) => actions.updateSettings({ boardPanelWidthPx }),
    [actions]
  );

  return (
    <Flex h="full" maxW="full" minH="0" minW="0" w="full">
      {boardPanelCollapsed ? null : (
        <>
          <Flex
            ref={boardPanelRef}
            flexShrink={0}
            minH="0"
            overflow="hidden"
            pb="2"
            pe="2"
            ps="2"
            pt={CHROME_INSET_PADDING_TOP}
            w={`${boardPanelWidthPx}px`}
          >
            <GalleryBoardsPanel />
          </Flex>
          <ResizeHandle
            label={t('widgets.gallery.resizeBoardPanel')}
            max={GALLERY_BOARD_PANEL_MAX_WIDTH_PX}
            min={GALLERY_BOARD_PANEL_MIN_WIDTH_PX}
            orientation="vertical"
            pane="before"
            paneRef={boardPanelRef}
            value={boardPanelWidthPx}
            onCommit={handleCommitWidth}
          />
        </>
      )}
      <Stack flex="1" gap="0" minH="0" minW="0">
        <HStack gap="1" minW="0" pb="2" pe="3" ps="3" pt={CHROME_INSET_PADDING_TOP}>
          <GalleryViewTabs idBase={viewTabsIdBase} />
          <Spacer />
          <Box flex="1" maxW="22rem" minW="9rem">
            <GalleryItemSearch />
          </Box>
          <GalleryStarredFilterToggle />
          <GalleryItemSortMenu />
          <GalleryUploadButton
            boards={gallery.boards}
            selectedBoardId={gallery.selectedBoardId}
            onUploadFiles={actions.uploadFiles}
          />
        </HStack>
        <Box
          aria-labelledby={segmentTabsTabId(viewTabsIdBase, gallery.galleryView)}
          flex="1"
          id={segmentTabsPanelId(viewTabsIdBase)}
          minH="0"
          minW="0"
          pb="2"
          pe="3"
          ps="3"
          role="tabpanel"
        >
          <GalleryImageGrid />
        </Box>
        <GallerySelectionBar />
      </Stack>
    </Flex>
  );
};

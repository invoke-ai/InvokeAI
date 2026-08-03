import { Box, Flex, HStack, Spacer, Stack } from '@chakra-ui/react';
import { GALLERY_BOARD_PANEL_MAX_WIDTH_PX, GALLERY_BOARD_PANEL_MIN_WIDTH_PX } from '@features/gallery/core/settings';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { GalleryBoardsPanel } from './GalleryBoardsPanel';
import { GalleryImageGrid } from './GalleryImageGrid';
import { GalleryItemSearch } from './GalleryItemSearch';
import { GalleryItemSortMenu } from './GalleryItemSortMenu';
import { GallerySelectionBar } from './GallerySelectionBar';
import { GallerySplitHandle } from './GallerySplitHandle';
import { GalleryViewTabs } from './GalleryViewTabs';
import { useGalleryWidget } from './GalleryWidgetContext';

const CHROME_INSET_PADDING_TOP = 'calc(var(--chakra-spacing-2) + var(--wb-center-chrome-inset, 0px))';

/**
 * Roomy arrangement: board panel beside the items area, always open, with the
 * header controls on one row. Pure arrangement — every child here is the same
 * component the stacked shell renders.
 */
export const GalleryWideLayout = () => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const { boardPanelCollapsed, boardPanelWidthPx } = gallery.settings;
  const [dragWidthPx, setDragWidthPx] = useState<number | null>(null);
  const displayWidthPx = dragWidthPx ?? boardPanelWidthPx;

  const handleCommitWidth = useCallback(
    (boardPanelWidthPx: number) => actions.updateSettings({ boardPanelWidthPx }),
    [actions]
  );

  return (
    <Flex h="full" maxW="full" minH="0" minW="0" w="full">
      {boardPanelCollapsed ? null : (
        <>
          <Flex
            flexShrink={0}
            minH="0"
            overflow="hidden"
            pb="2"
            pe="1"
            ps="2"
            pt={CHROME_INSET_PADDING_TOP}
            w={`${displayWidthPx}px`}
          >
            <GalleryBoardsPanel />
          </Flex>
          <GallerySplitHandle
            label={t('widgets.gallery.resizeBoardPanel')}
            max={GALLERY_BOARD_PANEL_MAX_WIDTH_PX}
            min={GALLERY_BOARD_PANEL_MIN_WIDTH_PX}
            orientation="vertical"
            sizePx={displayWidthPx}
            onCommit={handleCommitWidth}
            onPreview={setDragWidthPx}
          />
        </>
      )}
      <Stack flex="1" gap="0" minH="0" minW="0">
        {/* Tabs anchor the start of the row; everything you reach for while
            looking at the grid — find, order, add, configure — collects at the
            end, so the eye has one place to go. */}
        <HStack gap="1" minW="0" pb="2" pe="3" ps="3" pt={CHROME_INSET_PADDING_TOP}>
          <GalleryViewTabs />
          <Spacer />
          <Box flex="1" maxW="22rem" minW="9rem">
            <GalleryItemSearch />
          </Box>
          <GalleryItemSortMenu />
        </HStack>
        <Box flex="1" minH="0" minW="0" pb="2" pe="3" ps="3">
          <GalleryImageGrid />
        </Box>
        <GallerySelectionBar />
      </Stack>
    </Flex>
  );
};

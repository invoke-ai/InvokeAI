import { Flex, HStack, Spacer, Stack } from '@chakra-ui/react';
import {
  GALLERY_BOARD_PANEL_MAX_HEIGHT_PX,
  GALLERY_BOARD_PANEL_MIN_HEIGHT_PX,
  GALLERY_MIN_GRID_HEIGHT_PX,
} from '@features/gallery/core/settings';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { GalleryBoardsPanel } from './GalleryBoardsPanel';
import { GalleryImageGrid } from './GalleryImageGrid';
import { GalleryItemSearch } from './GalleryItemSearch';
import { GalleryItemSortMenu } from './GalleryItemSortMenu';
import { GallerySelectionBar } from './GallerySelectionBar';
import { GALLERY_SPLIT_HANDLE_SIZE_PX, GallerySplitHandle } from './GallerySplitHandle';
import { GalleryUploadButton } from './GalleryUploadButton';
import { GalleryViewTabs } from './GalleryViewTabs';
import { useGalleryWidget } from './GalleryWidgetContext';

/**
 * Narrow arrangement: board panel above the items area. Pure arrangement —
 * every child here is the same component the wide shell renders. The board
 * panel's disclosure lives in the widget frame's header label, not here.
 */
export const GalleryStackedLayout = () => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const { boardPanelCollapsed, boardPanelHeightPx } = gallery.settings;
  const [dragHeightPx, setDragHeightPx] = useState<number | null>(null);
  const [containerContentHeightPx, setContainerContentHeightPx] = useState<number | null>(null);
  const [controlsHeightPx, setControlsHeightPx] = useState<number | null>(null);
  const containerObserverRef = useRef<ResizeObserver | null>(null);
  const controlsObserverRef = useRef<ResizeObserver | null>(null);

  const containerRef = useCallback((element: HTMLDivElement | null) => {
    containerObserverRef.current?.disconnect();
    containerObserverRef.current = null;

    if (!element) {
      return;
    }

    const observer = new ResizeObserver(([entry]) => {
      if (entry) {
        setContainerContentHeightPx(entry.contentRect.height);
      }
    });

    observer.observe(element);
    containerObserverRef.current = observer;
  }, []);

  const controlsRef = useCallback((element: HTMLDivElement | null) => {
    controlsObserverRef.current?.disconnect();
    controlsObserverRef.current = null;

    if (!element) {
      return;
    }

    const observer = new ResizeObserver(([entry]) => {
      if (entry) {
        setControlsHeightPx(entry.contentRect.height);
      }
    });

    observer.observe(element);
    controlsObserverRef.current = observer;
  }, []);

  const measuredMaximumPx = useMemo(() => {
    if (containerContentHeightPx === null || controlsHeightPx === null) {
      return GALLERY_BOARD_PANEL_MIN_HEIGHT_PX;
    }

    const availableHeightPx =
      containerContentHeightPx - controlsHeightPx - GALLERY_SPLIT_HANDLE_SIZE_PX - 3 * 8 - GALLERY_MIN_GRID_HEIGHT_PX;

    return Math.min(
      GALLERY_BOARD_PANEL_MAX_HEIGHT_PX,
      Math.max(GALLERY_BOARD_PANEL_MIN_HEIGHT_PX, Math.floor(availableHeightPx))
    );
  }, [containerContentHeightPx, controlsHeightPx]);
  const displayHeightPx = Math.min(dragHeightPx ?? boardPanelHeightPx, measuredMaximumPx);

  const handleCommitHeight = useCallback(
    (boardPanelHeightPx: number) => actions.updateSettings({ boardPanelHeightPx }),
    [actions]
  );

  return (
    <Stack gap="0" h="full" maxW="full" minH="0" minW="0" w="full">
      <Stack
        ref={containerRef}
        flex="1"
        gap="2"
        minH="0"
        minW="0"
        pb="2"
        pt="calc(var(--chakra-spacing-2) + var(--wb-center-chrome-inset, 0px))"
        px="2"
      >
        {boardPanelCollapsed ? null : (
          <>
            {/* A flex box with a definite height and hidden overflow: the panel
                sizes its own scroll area from this, and without the containment
                its board list spills over the grid below. */}
            <Flex flexShrink={0} h={`${displayHeightPx}px`} minH="0" overflow="hidden" w="full">
              <GalleryBoardsPanel />
            </Flex>
            <GallerySplitHandle
              label={t('widgets.gallery.resizeBoardPanel')}
              max={measuredMaximumPx}
              min={GALLERY_BOARD_PANEL_MIN_HEIGHT_PX}
              orientation="horizontal"
              sizePx={displayHeightPx}
              onCommit={handleCommitHeight}
              onPreview={setDragHeightPx}
            />
          </>
        )}
        <Stack ref={controlsRef} flexShrink={0} gap="2" minW="0">
          <HStack gap="1" minW="0">
            <GalleryViewTabs />
            <Spacer />
            <GalleryItemSortMenu />
            <GalleryUploadButton
              boards={gallery.boards}
              selectedBoardId={gallery.selectedBoardId}
              onUploadFiles={actions.uploadFiles}
            />
          </HStack>
          <GalleryItemSearch />
        </Stack>
        <Flex data-gallery-grid-wrapper flex="1" minH={`${GALLERY_MIN_GRID_HEIGHT_PX}px`} minW="0">
          <GalleryImageGrid />
        </Flex>
      </Stack>
      <GallerySelectionBar />
    </Stack>
  );
};

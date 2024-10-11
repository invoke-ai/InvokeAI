import { Box, Button, Collapse, Divider, Flex, IconButton, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion } from 'common/hooks/focus';
import { GalleryHeader } from 'features/gallery/components/GalleryHeader';
import { selectBoardSearchText } from 'features/gallery/store/gallerySelectors';
import { boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { usePanel, type UsePanelOptions } from 'features/ui/hooks/usePanel';
import type { CSSProperties } from 'react';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiCaretUpBold, PiMagnifyingGlassBold } from 'react-icons/pi';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';
import { Panel, PanelGroup } from 'react-resizable-panels';

import BoardsListWrapper from './Boards/BoardsList/BoardsListWrapper';
import BoardsSearch from './Boards/BoardsList/BoardsSearch';
import BoardsSettingsPopover from './Boards/BoardsSettingsPopover';
import { Gallery } from './Gallery';

const COLLAPSE_STYLES: CSSProperties = { flexShrink: 0, minHeight: 0 };

const GalleryPanelContent = () => {
  const { t } = useTranslation();
  const boardSearchText = useAppSelector(selectBoardSearchText);
  const dispatch = useAppDispatch();
  const boardSearchDisclosure = useDisclosure({ defaultIsOpen: !!boardSearchText.length });
  const imperativePanelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  const galleryPanelFocusRef = useRef<HTMLDivElement>(null);
  useFocusRegion('gallery', galleryPanelFocusRef);

  const boardsListPanelOptions = useMemo<UsePanelOptions>(
    () => ({
      id: 'boards-list-panel',
      minSizePx: 128,
      defaultSizePx: 256,
      imperativePanelGroupRef,
      panelGroupDirection: 'vertical',
    }),
    []
  );
  const boardsListPanel = usePanel(boardsListPanelOptions);

  const handleClickBoardSearch = useCallback(() => {
    if (boardSearchText.length) {
      dispatch(boardSearchTextChanged(''));
    }
    boardSearchDisclosure.onToggle();
    boardsListPanel.expand();
  }, [boardSearchText.length, boardSearchDisclosure, boardsListPanel, dispatch]);

  return (
    <Flex ref={galleryPanelFocusRef} position="relative" flexDirection="column" h="full" w="full" tabIndex={-1}>
      <Flex alignItems="center" w="full">
        <Flex w="25%">
          <Button
            size="sm"
            variant="ghost"
            onClick={boardsListPanel.toggle}
            rightIcon={boardsListPanel.isCollapsed ? <PiCaretDownBold /> : <PiCaretUpBold />}
          >
            {boardsListPanel.isCollapsed ? t('boards.viewBoards') : t('boards.hideBoards')}
          </Button>
        </Flex>
        <GalleryHeader />
        <Flex h="full" w="25%" justifyContent="flex-end">
          <BoardsSettingsPopover />
          <IconButton
            size="sm"
            variant="link"
            alignSelf="stretch"
            onClick={handleClickBoardSearch}
            tooltip={
              boardSearchDisclosure.isOpen ? `${t('gallery.exitBoardSearch')}` : `${t('gallery.displayBoardSearch')}`
            }
            aria-label={t('gallery.displayBoardSearch')}
            icon={<PiMagnifyingGlassBold />}
            colorScheme={boardSearchDisclosure.isOpen ? 'invokeBlue' : 'base'}
          />
        </Flex>
      </Flex>

      <PanelGroup ref={imperativePanelGroupRef} direction="vertical" autoSaveId="boards-list-panel">
        <Panel collapsible {...boardsListPanel.panelProps}>
          <Flex flexDir="column" w="full" h="full">
            <Collapse in={boardSearchDisclosure.isOpen} style={COLLAPSE_STYLES}>
              <Box w="full" pt={2}>
                <BoardsSearch />
              </Box>
            </Collapse>
            <Divider pt={2} />
            <BoardsListWrapper />
          </Flex>
        </Panel>
        <ResizeHandle id="gallery-panel-handle" {...boardsListPanel.resizeHandleProps} />
        <Panel id="gallery-wrapper-panel" minSize={20}>
          <Gallery />
        </Panel>
      </PanelGroup>
    </Flex>
  );
};

export default memo(GalleryPanelContent);

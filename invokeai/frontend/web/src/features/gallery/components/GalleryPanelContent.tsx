import { Box, Button, Collapse, Divider, Flex, IconButton, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useScopeOnFocus } from 'common/hooks/interactionScopes';
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
import { Gallery } from './Gallery';
import GallerySettingsPopover from './GallerySettingsPopover/GallerySettingsPopover';

const COLLAPSE_STYLES: CSSProperties = { flexShrink: 0, minHeight: 0 };

const GalleryPanelContent = () => {
  const { t } = useTranslation();
  const boardSearchText = useAppSelector(selectBoardSearchText);
  const dispatch = useAppDispatch();
  const boardSearchDisclosure = useDisclosure({ defaultIsOpen: !!boardSearchText.length });
  const imperativePanelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  const ref = useRef<HTMLDivElement>(null);
  useScopeOnFocus('gallery', ref);

  const boardsListPanelOptions = useMemo<UsePanelOptions>(
    () => ({
      id: 'boards-list-panel',
      minSize: 128,
      defaultSize: 208,
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
    <Flex ref={ref} position="relative" flexDirection="column" h="full" w="full" tabIndex={-1}>
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
          <GallerySettingsPopover />
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
        <ResizeHandle id="gallery-panel-handle" orientation="horizontal" {...boardsListPanel.resizeHandleProps} />
        <Panel id="gallery-wrapper-panel" minSize={20}>
          <Gallery />
        </Panel>
      </PanelGroup>
    </Flex>
  );
};

export default memo(GalleryPanelContent);

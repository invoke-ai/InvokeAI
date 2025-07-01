import { Box, Button, Collapse, Divider, Flex, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useDisclosure } from 'common/hooks/useBoolean';
import { BoardsListWrapper } from 'features/gallery/components/Boards/BoardsList/BoardsListWrapper';
import { BoardsSearch } from 'features/gallery/components/Boards/BoardsList/BoardsSearch';
import { BoardsSettingsPopover } from 'features/gallery/components/Boards/BoardsSettingsPopover';
import { GalleryHeader } from 'features/gallery/components/GalleryHeader';
import { selectBoardSearchText } from 'features/gallery/store/gallerySelectors';
import { boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import { useAutoLayoutContext } from 'features/ui/layouts/auto-layout-context';
import { BOARD_PANEL_DEFAULT_HEIGHT_PX, BOARD_PANEL_MIN_HEIGHT_PX, BOARDS_PANEL_ID } from 'features/ui/layouts/shared';
import { useCollapsibleGridviewPanel } from 'features/ui/layouts/use-collapsible-gridview-panel';
import type { CSSProperties } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiCaretUpBold, PiMagnifyingGlassBold } from 'react-icons/pi';

const COLLAPSE_STYLES: CSSProperties = { flexShrink: 0, minHeight: 0 };

export const BoardsPanel = memo(() => {
  const boardSearchText = useAppSelector(selectBoardSearchText);
  const searchDisclosure = useDisclosure(!!boardSearchText);
  const { _$rightPanelApi } = useAutoLayoutContext();
  const gridviewPanelApi = useStore(_$rightPanelApi);
  const collapsibleApi = useCollapsibleGridviewPanel(
    gridviewPanelApi,
    BOARDS_PANEL_ID,
    'vertical',
    BOARD_PANEL_DEFAULT_HEIGHT_PX,
    BOARD_PANEL_MIN_HEIGHT_PX
  );
  const isCollapsed = useStore(collapsibleApi.$isCollapsed);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onClickBoardSearch = useCallback(() => {
    if (boardSearchText.length) {
      dispatch(boardSearchTextChanged(''));
    }
    if (!searchDisclosure.isOpen && collapsibleApi.$isCollapsed.get()) {
      collapsibleApi.expand();
    }
    searchDisclosure.toggle();
  }, [boardSearchText.length, searchDisclosure, collapsibleApi, dispatch]);

  return (
    <Flex flexDir="column" w="full" h="full">
      <Flex alignItems="center" justifyContent="space-between" w="full">
        <Flex flexGrow={1} flexBasis={0}>
          <Button
            size="sm"
            variant="ghost"
            onClick={collapsibleApi.toggle}
            leftIcon={isCollapsed ? <PiCaretDownBold /> : <PiCaretUpBold />}
          >
            Boards
          </Button>
        </Flex>
        <Flex>
          <GalleryHeader />
        </Flex>
        <Flex flexGrow={1} flexBasis={0} justifyContent="flex-end">
          <BoardsSettingsPopover />
          <IconButton
            size="sm"
            variant="link"
            alignSelf="stretch"
            onClick={onClickBoardSearch}
            tooltip={searchDisclosure.isOpen ? `${t('gallery.exitBoardSearch')}` : `${t('gallery.displayBoardSearch')}`}
            aria-label={t('gallery.displayBoardSearch')}
            icon={<PiMagnifyingGlassBold />}
            colorScheme={searchDisclosure.isOpen ? 'invokeBlue' : 'base'}
          />
        </Flex>
      </Flex>
      <Collapse in={searchDisclosure.isOpen} style={COLLAPSE_STYLES}>
        <Box w="full" pt={2}>
          <BoardsSearch />
        </Box>
      </Collapse>
      <Divider pt={2} />
      <BoardsListWrapper />
    </Flex>
  );
});
BoardsPanel.displayName = 'BoardsPanel';

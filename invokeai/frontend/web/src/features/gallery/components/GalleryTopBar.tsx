import { Button, Flex, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useBoardSearchDisclosure } from 'features/gallery/components/Boards/BoardsList/BoardsSearch';
import { BoardsSettingsPopover } from 'features/gallery/components/Boards/BoardsSettingsPopover';
import { GalleryHeader } from 'features/gallery/components/GalleryHeader';
import { selectBoardSearchText } from 'features/gallery/store/gallerySelectors';
import { boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import { useAutoLayoutContext } from 'features/ui/layouts/auto-layout-context';
import { BOARD_PANEL_DEFAULT_HEIGHT_PX, BOARD_PANEL_MIN_HEIGHT_PX, BOARDS_PANEL_ID } from 'features/ui/layouts/shared';
import { useCollapsibleGridviewPanel } from 'features/ui/layouts/use-collapsible-gridview-panel';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiCaretUpBold, PiMagnifyingGlassBold } from 'react-icons/pi';

export const GalleryTopBar = memo(() => {
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
  const boardSearchText = useAppSelector(selectBoardSearchText);
  const boardSearchDisclosure = useBoardSearchDisclosure();

  const onClickBoardSearch = useCallback(() => {
    if (boardSearchText.length) {
      dispatch(boardSearchTextChanged(''));
    }
    if (!boardSearchDisclosure.isOpen && collapsibleApi.$isCollapsed.get()) {
      collapsibleApi.expand();
    }
    boardSearchDisclosure.toggle();
  }, [boardSearchText.length, boardSearchDisclosure, collapsibleApi, dispatch]);

  return (
    <Flex alignItems="center" justifyContent="space-between" w="full">
      <Flex flexGrow={1} flexBasis={0}>
        <Button
          size="sm"
          variant="ghost"
          onClick={collapsibleApi.toggle}
          rightIcon={isCollapsed ? <PiCaretDownBold /> : <PiCaretUpBold />}
        >
          {isCollapsed ? t('boards.viewBoards') : t('boards.hideBoards')}
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
          tooltip={
            boardSearchDisclosure.isOpen ? `${t('gallery.exitBoardSearch')}` : `${t('gallery.displayBoardSearch')}`
          }
          aria-label={t('gallery.displayBoardSearch')}
          icon={<PiMagnifyingGlassBold />}
          colorScheme={boardSearchDisclosure.isOpen ? 'invokeBlue' : 'base'}
        />
      </Flex>
    </Flex>
  );
});
GalleryTopBar.displayName = 'GalleryTopBar';

import type { UseDisclosureReturn } from '@invoke-ai/ui-library';
import { Button, Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { BoardsSettingsPopover } from 'features/gallery/components/Boards/BoardsSettingsPopover';
import { GalleryHeader } from 'features/gallery/components/GalleryHeader';
import { selectBoardSearchText } from 'features/gallery/store/gallerySelectors';
import { boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import type { UsePanelReturn } from 'features/ui/hooks/usePanel';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiCaretUpBold, PiMagnifyingGlassBold } from 'react-icons/pi';

export const GalleryTopBar = memo(
  ({
    boardsListPanel,
    boardSearchDisclosure,
  }: {
    boardsListPanel: UsePanelReturn;
    boardSearchDisclosure: UseDisclosureReturn;
  }) => {
    const { t } = useTranslation();
    const dispatch = useAppDispatch();
    const boardSearchText = useAppSelector(selectBoardSearchText);

    const onClickBoardSearch = useCallback(() => {
      if (boardSearchText.length) {
        dispatch(boardSearchTextChanged(''));
      }
      boardSearchDisclosure.onToggle();
      boardsListPanel.expand();
    }, [boardSearchText.length, boardSearchDisclosure, boardsListPanel, dispatch]);

    return (
      <Flex alignItems="center" justifyContent="space-between" w="full">
        <Flex flexGrow={1} flexBasis={0}>
          <Button
            size="sm"
            variant="ghost"
            onClick={boardsListPanel.toggle}
            rightIcon={boardsListPanel.isCollapsed ? <PiCaretDownBold /> : <PiCaretUpBold />}
          >
            {boardsListPanel.isCollapsed ? t('boards.viewBoards') : t('boards.hideBoards')}
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
  }
);
GalleryTopBar.displayName = 'GalleryTopBar';

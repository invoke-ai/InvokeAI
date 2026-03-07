import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { fixTooltipCloseOnScrollStyles } from 'common/util/fixTooltipCloseOnScrollStyles';
import {
  selectBoardSearchText,
  selectListBoardsQueryArgs,
  selectSelectedBoardId,
} from 'features/gallery/store/gallerySelectors';
import { memo, useCallback,useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMagnifyingGlassBold, PiSidebarSimpleBold } from 'react-icons/pi';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

import { AddBoardIconButton } from './AddBoardButton';
import BoardItem from './BoardItem';
import { BoardsSearch } from './BoardsSearch';
import { BoardsSettingsPopover } from './BoardsSettingsPopover';

type BoardsListProps = {
  isCollapsed?: boolean;
  onCollapseBoards?: () => void;
  onExpandBoards?: () => void;
  showHeaderAddButton?: boolean;
};

export const BoardsList = memo(({ onCollapseBoards, onExpandBoards, isCollapsed }: BoardsListProps) => {
  const { t } = useTranslation();
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const boardSearchText = useAppSelector(selectBoardSearchText);
  const queryArgs = useAppSelector(selectListBoardsQueryArgs);
  const { data: boards } = useListAllBoardsQuery(queryArgs);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const filteredBoards = useMemo(() => {
    if (!boards) {
      return EMPTY_ARRAY;
    }

    if (boardSearchText.length) {
      return boards.filter((board) => board.board_name.toLowerCase().includes(boardSearchText.toLowerCase()));
    }

    return boards;
  }, [boardSearchText, boards]);

  const boardElements = useMemo(() => {
    const elements = [];

    if (!boardSearchText.length) {
      elements.push(<BoardItem key="none" board={null} isSelected={selectedBoardId === 'none'} isCollapsed={isCollapsed} />);
    }

    filteredBoards.forEach((board) => {
      elements.push(<BoardItem board={board} isSelected={selectedBoardId === board.board_id} key={board.board_id} isCollapsed={isCollapsed} />);
    });

    return elements;
  }, [boardSearchText.length, filteredBoards, selectedBoardId, isCollapsed]);

  const onFocusSearch = useCallback(() => {
    if (onExpandBoards) {
      onExpandBoards();
    }
    setTimeout(() => {
      searchInputRef.current?.focus();
    }, 0)
  }, [onExpandBoards]);

  return (
    <Flex direction="column">
      {isCollapsed ? (
        <>
          <Flex flexDir="column" pb={1} pt={2} gap={1}>
            <IconButton
              icon={<PiSidebarSimpleBold />}
              onClick={onExpandBoards}
              tooltip={t('gallery.showBoardsSidebar')}
              aria-label={t('gallery.showBoardsSidebar')}
              variant="ghost"
            />
            <IconButton
              icon={<PiMagnifyingGlassBold />}
              onClick={onFocusSearch}
              tooltip={t('boards.searchBoard')}
              aria-label={t('boards.searchBoard')}
              variant="ghost"
            />
          </Flex>
          <Flex direction="column" gap={1} style={fixTooltipCloseOnScrollStyles}>
            {boardElements.length ? (
              boardElements
            ) : (
              <Text variant="subtext" textAlign="center">
                {t('boards.noBoards', { boardType: boardSearchText.length ? 'Matching' : '' })}
              </Text>
            )}
          </Flex>
        </>
      ) : (
        <>
          <Flex
            position="sticky"
            w="full"
            justifyContent="space-between"
            alignItems="center"
            py={1}
            zIndex={1}
            top={0}
            bg="base.900"
            h={12}
          >
            <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.400" px={1}>
              {t('boards.boards')}
            </Text>
            <Flex alignItems="center">
              <IconButton
                size="sm"
                variant="link"
                icon={<PiSidebarSimpleBold />}
                onClick={onCollapseBoards}
                tooltip={t('gallery.hideBoardsSidebar')}
                aria-label={t('gallery.toggleBoardsSidebar')}
              />
              <BoardsSettingsPopover />
              <AddBoardIconButton />
            </Flex>
          </Flex>
          <Flex pb={2}>
            <BoardsSearch ref={searchInputRef} />
          </Flex>
          <Flex direction="column" gap={1} style={fixTooltipCloseOnScrollStyles}>
            {boardElements.length ? (
              boardElements
            ) : (
              <Text variant="subtext" textAlign="center">
                {t('boards.noBoards', { boardType: boardSearchText.length ? 'Matching' : '' })}
              </Text>
            )}
          </Flex>
        </>
      )}
    </Flex>
  )
});
BoardsList.displayName = 'BoardsList';

import { Collapse, Flex, IconButton, Text, useDisclosure } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { fixTooltipCloseOnScrollStyles } from 'common/util/fixTooltipCloseOnScrollStyles';
import {
  selectBoardSearchText,
  selectListBoardsQueryArgs,
  selectSelectedBoardId,
} from 'features/gallery/store/gallerySelectors';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSidebarSimpleBold } from 'react-icons/pi';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

import { AddBoardIconButton } from './AddBoardButton';
import BoardItem from './BoardItem';
import { BoardsSettingsPopover } from './BoardsSettingsPopover';
import { BoardsSearch } from './BoardsSearch';

const HEADER_ACTION_BUTTON_SIZE = 10;

type BoardsListProps = {
  onCollapse?: () => void;
  showHeaderAddButton?: boolean;
};

export const BoardsList = memo(({ onCollapse, showHeaderAddButton = false }: BoardsListProps) => {
  const { t } = useTranslation();
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const boardSearchText = useAppSelector(selectBoardSearchText);
  const queryArgs = useAppSelector(selectListBoardsQueryArgs);
  const { data: boards } = useListAllBoardsQuery(queryArgs);
  const { isOpen } = useDisclosure({ defaultIsOpen: true });

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
      elements.push(<BoardItem key="none" board={null} isSelected={selectedBoardId === 'none'} />);
    }

    filteredBoards.forEach((board) => {
      elements.push(<BoardItem board={board} isSelected={selectedBoardId === board.board_id} key={board.board_id} />);
    });

    return elements;
  }, [boardSearchText.length, filteredBoards, selectedBoardId]);

  return (
    <Flex direction="column">
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
        <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.400">
          {t('boards.boards')}
        </Text>
        <Flex alignItems="center" gap={1}>
          {onCollapse && (
            <IconButton
              size="sm"
              variant="ghost"
              icon={<PiSidebarSimpleBold />}
              onClick={onCollapse}
              tooltip={t('gallery.hideBoardsSidebar')}
              aria-label={t('gallery.toggleBoardsSidebar')}
              h={HEADER_ACTION_BUTTON_SIZE}
              w={HEADER_ACTION_BUTTON_SIZE}
              p={0}
            />
          )}
          <BoardsSettingsPopover h={HEADER_ACTION_BUTTON_SIZE} w={HEADER_ACTION_BUTTON_SIZE} p={0} />
          {showHeaderAddButton && <AddBoardIconButton h={HEADER_ACTION_BUTTON_SIZE} w={HEADER_ACTION_BUTTON_SIZE} p={0} />}
        </Flex>
      </Flex>
      <Flex pb={2}>
        <BoardsSearch />
      </Flex>
      <Collapse in={isOpen} style={fixTooltipCloseOnScrollStyles}>
        <Flex direction="column">
          {boardElements.length ? (
            boardElements
          ) : (
            <Text variant="subtext" textAlign="center">
              {t('boards.noBoards', { boardType: boardSearchText.length ? 'Matching' : '' })}
            </Text>
          )}
        </Flex>
      </Collapse>
    </Flex>
  );
});
BoardsList.displayName = 'BoardsList';

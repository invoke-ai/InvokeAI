import { Collapse, Flex, Text, useDisclosure } from '@invoke-ai/ui-library';
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
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

import AddBoardButton from './AddBoardButton';
import GalleryBoard from './GalleryBoard';
import NoBoardBoard from './NoBoardBoard';

export const BoardsList = memo(() => {
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
      elements.push(<NoBoardBoard key="none" isSelected={selectedBoardId === 'none'} />);
    }

    filteredBoards.forEach((board) => {
      elements.push(
        <GalleryBoard board={board} isSelected={selectedBoardId === board.board_id} key={board.board_id} />
      );
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
        ps={2}
        py={1}
        zIndex={1}
        top={0}
        bg="base.900"
      >
        <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.500">
          {t('boards.boards')}
        </Text>
        <AddBoardButton />
      </Flex>
      <Collapse in={isOpen} style={fixTooltipCloseOnScrollStyles}>
        <Flex direction="column" gap={1}>
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

import { Button, Collapse, Flex, Icon, Text, useDisclosure } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectBoardSearchText,
  selectListBoardsQueryArgs,
  selectSelectedBoardId,
} from 'features/gallery/store/gallerySelectors';
import { selectAllowPrivateBoards } from 'features/system/store/configSelectors';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

import AddBoardButton from './AddBoardButton';
import GalleryBoard from './GalleryBoard';
import NoBoardBoard from './NoBoardBoard';

type Props = {
  isPrivate: boolean;
};

export const BoardsList = ({ isPrivate }: Props) => {
  const { t } = useTranslation();
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const boardSearchText = useAppSelector(selectBoardSearchText);
  const queryArgs = useAppSelector(selectListBoardsQueryArgs);
  const { data: boards } = useListAllBoardsQuery(queryArgs);
  const allowPrivateBoards = useAppSelector(selectAllowPrivateBoards);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });

  const filteredBoards = useMemo(() => {
    if (!boards) {
      return EMPTY_ARRAY;
    }

    return boards.filter((board) => {
      if (boardSearchText.length) {
        return board.is_private === isPrivate && board.board_name.toLowerCase().includes(boardSearchText.toLowerCase());
      } else {
        return board.is_private === isPrivate;
      }
    });
  }, [boardSearchText, boards, isPrivate]);

  const boardElements = useMemo(() => {
    const elements = [];
    if (allowPrivateBoards && isPrivate && !boardSearchText.length) {
      elements.push(<NoBoardBoard key="none" isSelected={selectedBoardId === 'none'} />);
    }

    if (!allowPrivateBoards && !boardSearchText.length) {
      elements.push(<NoBoardBoard key="none" isSelected={selectedBoardId === 'none'} />);
    }

    filteredBoards.forEach((board) => {
      elements.push(
        <GalleryBoard board={board} isSelected={selectedBoardId === board.board_id} key={board.board_id} />
      );
    });

    return elements;
  }, [allowPrivateBoards, isPrivate, boardSearchText.length, filteredBoards, selectedBoardId]);

  const boardListTitle = useMemo(() => {
    if (allowPrivateBoards) {
      return isPrivate ? t('boards.private') : t('boards.shared');
    } else {
      return t('boards.boards');
    }
  }, [isPrivate, allowPrivateBoards, t]);

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
        {allowPrivateBoards ? (
          <Button variant="unstyled" onClick={onToggle}>
            <Flex gap="2" alignItems="center">
              <Icon
                boxSize={4}
                as={PiCaretDownBold}
                transform={isOpen ? undefined : 'rotate(-90deg)'}
                fill="base.500"
              />
              <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.500">
                {boardListTitle}
              </Text>
            </Flex>
          </Button>
        ) : (
          <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.500">
            {boardListTitle}
          </Text>
        )}
        <AddBoardButton isPrivateBoard={isPrivate} />
      </Flex>
      <Collapse in={isOpen}>
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
};

import { Flex, Text } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import DeleteBoardModal from 'features/gallery/components/Boards/DeleteBoardModal';
import { selectListBoardsQueryArgs } from 'features/gallery/store/gallerySelectors';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import type { BoardDTO } from 'services/api/types';

import AddBoardButton from './AddBoardButton';
import GalleryBoard from './GalleryBoard';
import NoBoardBoard from './NoBoardBoard';

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

const BoardsList = () => {
  const selectedBoardId = useAppSelector((s) => s.gallery.selectedBoardId);
  const boardSearchText = useAppSelector((s) => s.gallery.boardSearchText);
  const allowPrivateBoards = useAppSelector((s) => s.config.allowPrivateBoards);
  const queryArgs = useAppSelector(selectListBoardsQueryArgs);
  const { data: boards } = useListAllBoardsQuery(queryArgs);
  const [boardToDelete, setBoardToDelete] = useState<BoardDTO>();
  const { t } = useTranslation();

  const { filteredPrivateBoards, filteredSharedBoards } = useMemo(() => {
    const filteredBoards = boardSearchText
      ? boards?.filter((board) => board.board_name.toLowerCase().includes(boardSearchText.toLowerCase()))
      : boards;
    const filteredPrivateBoards = filteredBoards?.filter((board) => board.is_private) ?? EMPTY_ARRAY;
    const filteredSharedBoards = filteredBoards?.filter((board) => !board.is_private) ?? EMPTY_ARRAY;
    return { filteredPrivateBoards, filteredSharedBoards };
  }, [boardSearchText, boards]);

  return (
    <>
      <Flex flexDir="column" gap={2} borderRadius="base" maxHeight="100%">
        <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
          {allowPrivateBoards && (
            <>
              <Flex w="full" justifyContent="space-between" alignItems="center" ps={2}>
                <Text fontSize="md" fontWeight="medium" userSelect="none">
                  {t('boards.private')}
                </Text>
                <AddBoardButton isPrivateBoard={true} />
              </Flex>
              <Flex direction="column" maxH={100} gap={1}>
                <NoBoardBoard isSelected={selectedBoardId === 'none'} />
                {filteredPrivateBoards.map((board) => (
                  <GalleryBoard
                    board={board}
                    isSelected={selectedBoardId === board.board_id}
                    setBoardToDelete={setBoardToDelete}
                    key={board.board_id}
                  />
                ))}
              </Flex>
            </>
          )}
          <Flex w="full" justifyContent="space-between" alignItems="center" ps={2}>
            <Text fontSize="md" fontWeight="medium" userSelect="none">
              {allowPrivateBoards ? t('boards.shared') : t('boards.boards')}
            </Text>
            <AddBoardButton isPrivateBoard={false} />
          </Flex>
          <Flex direction="column" gap={1}>
            {!allowPrivateBoards && <NoBoardBoard isSelected={selectedBoardId === 'none'} />}
            {filteredSharedBoards.map((board) => (
              <GalleryBoard
                board={board}
                isSelected={selectedBoardId === board.board_id}
                setBoardToDelete={setBoardToDelete}
                key={board.board_id}
              />
            ))}
          </Flex>
        </OverlayScrollbarsComponent>
      </Flex>
      <DeleteBoardModal boardToDelete={boardToDelete} setBoardToDelete={setBoardToDelete} />
    </>
  );
};
export default memo(BoardsList);

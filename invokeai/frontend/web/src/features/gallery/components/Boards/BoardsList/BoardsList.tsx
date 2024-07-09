import { Box, Collapse, Flex, Icon, Text, useDisclosure } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import DeleteBoardModal from 'features/gallery/components/Boards/DeleteBoardModal';
import GallerySettingsPopover from 'features/gallery/components/GallerySettingsPopover/GallerySettingsPopover';
import { selectListBoardsQueryArgs } from 'features/gallery/store/gallerySelectors';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretUpBold } from 'react-icons/pi';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import type { BoardDTO } from 'services/api/types';

import AddBoardButton from './AddBoardButton';
import BoardsSearch from './BoardsSearch';
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
  const privateBoardsDisclosure = useDisclosure({ defaultIsOpen: true });
  const sharedBoardsDisclosure = useDisclosure({ defaultIsOpen: true });
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
      <Flex layerStyle="first" flexDir="column" gap={2} p={2} mt={2} borderRadius="base">
        <Flex gap={2} alignItems="center">
          <BoardsSearch />
          <GallerySettingsPopover />
        </Flex>
        <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
          <Box maxH={346}>
            {allowPrivateBoards && (
              <>
                <Flex borderBottom="1px" borderColor="base.400" my="2" justifyContent="space-between">
                  <Flex onClick={privateBoardsDisclosure.onToggle} gap={2} alignItems="center" cursor="pointer">
                    <Icon
                      as={PiCaretUpBold}
                      boxSize={6}
                      transform={privateBoardsDisclosure.isOpen ? 'rotate(0deg)' : 'rotate(180deg)'}
                      transitionProperty="common"
                      transitionDuration="normal"
                      color="base.400"
                    />
                    <Text fontSize="md" fontWeight="medium">
                      {t('boards.private')}
                    </Text>
                  </Flex>
                  <AddBoardButton isPrivateBoard={true} />
                </Flex>
                <Collapse in={privateBoardsDisclosure.isOpen} animateOpacity>
                  <Flex direction="column">
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
                </Collapse>
              </>
            )}
            <Flex borderBottom="1px" borderColor="base.400" my="2" justifyContent="space-between">
              <Flex onClick={sharedBoardsDisclosure.onToggle} gap={2} alignItems="center" cursor="pointer">
                <Icon
                  as={PiCaretUpBold}
                  boxSize={6}
                  transform={sharedBoardsDisclosure.isOpen ? 'rotate(0deg)' : 'rotate(180deg)'}
                  transitionProperty="common"
                  transitionDuration="normal"
                  color="base.400"
                />
                <Text fontSize="md" fontWeight="medium">
                  {allowPrivateBoards ? t('boards.shared') : t('boards.boards')}
                </Text>
              </Flex>
              <AddBoardButton isPrivateBoard={false} />
            </Flex>
            <Collapse in={sharedBoardsDisclosure.isOpen} animateOpacity>
              <Flex direction="column">
                {filteredSharedBoards.map((board) => (
                  <GalleryBoard
                    board={board}
                    isSelected={selectedBoardId === board.board_id}
                    setBoardToDelete={setBoardToDelete}
                    key={board.board_id}
                  />
                ))}
              </Flex>
            </Collapse>
          </Box>
        </OverlayScrollbarsComponent>
      </Flex>
      <DeleteBoardModal boardToDelete={boardToDelete} setBoardToDelete={setBoardToDelete} />
    </>
  );
};

export default memo(BoardsList);

import {
  Badge,
  Box,
  ChakraProps,
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  Image,
  Text,
  useColorMode,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { MoveBoardDropData } from 'app/components/ImageDnd/typesafeDnd';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDroppable from 'common/components/IAIDroppable';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { FaUser } from 'react-icons/fa';
import { useUpdateBoardMutation } from 'services/api/endpoints/boards';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { BoardDTO } from 'services/api/types';
import { mode } from 'theme/util/mode';
import BoardContextMenu from '../BoardContextMenu';

const AUTO_ADD_BADGE_STYLES: ChakraProps['sx'] = {
  bg: 'accent.200',
  color: 'blackAlpha.900',
};

const BASE_BADGE_STYLES: ChakraProps['sx'] = {
  bg: 'base.500',
  color: 'whiteAlpha.900',
};
interface GalleryBoardProps {
  board: BoardDTO;
  isSelected: boolean;
  setBoardToDelete: (board?: BoardDTO) => void;
}

const GalleryBoard = memo(
  ({ board, isSelected, setBoardToDelete }: GalleryBoardProps) => {
    const dispatch = useAppDispatch();
    const selector = useMemo(
      () =>
        createSelector(
          stateSelector,
          ({ gallery }) => {
            const isSelectedForAutoAdd =
              board.board_id === gallery.autoAddBoardId;

            return { isSelectedForAutoAdd };
          },
          defaultSelectorOptions
        ),
      [board.board_id]
    );

    const { isSelectedForAutoAdd } = useAppSelector(selector);

    const { currentData: coverImage } = useGetImageDTOQuery(
      board.cover_image_name ?? skipToken
    );

    const { colorMode } = useColorMode();
    const { board_name, board_id } = board;
    const handleSelectBoard = useCallback(() => {
      dispatch(boardIdSelected(board_id));
    }, [board_id, dispatch]);

    const [updateBoard, { isLoading: isUpdateBoardLoading }] =
      useUpdateBoardMutation();

    const handleUpdateBoardName = (newBoardName: string) => {
      updateBoard({ board_id, changes: { board_name: newBoardName } });
    };

    const droppableData: MoveBoardDropData = useMemo(
      () => ({
        id: board_id,
        actionType: 'MOVE_BOARD',
        context: { boardId: board_id },
      }),
      [board_id]
    );

    return (
      <Box sx={{ touchAction: 'none', height: 'full' }}>
        <BoardContextMenu
          board={board}
          board_id={board_id}
          setBoardToDelete={setBoardToDelete}
        >
          {(ref) => (
            <Flex
              key={board_id}
              userSelect="none"
              ref={ref}
              sx={{
                flexDir: 'column',
                justifyContent: 'space-between',
                alignItems: 'center',
                cursor: 'pointer',
                w: 'full',
                h: 'full',
              }}
            >
              <Flex
                onClick={handleSelectBoard}
                sx={{
                  position: 'relative',
                  justifyContent: 'center',
                  alignItems: 'center',
                  borderRadius: 'base',
                  w: 'full',
                  aspectRatio: '1/1',
                  overflow: 'hidden',
                  shadow: isSelected ? 'selected.light' : undefined,
                  _dark: { shadow: isSelected ? 'selected.dark' : undefined },
                  flexShrink: 0,
                }}
              >
                {board.cover_image_name && coverImage?.thumbnail_url && (
                  <Image src={coverImage?.thumbnail_url} draggable={false} />
                )}
                {!(board.cover_image_name && coverImage?.thumbnail_url) && (
                  <IAINoContentFallback
                    boxSize={8}
                    icon={FaUser}
                    sx={{
                      borderWidth: '2px',
                      borderStyle: 'solid',
                      borderColor: 'base.200',
                      _dark: {
                        borderColor: 'base.800',
                      },
                    }}
                  />
                )}
                <Flex
                  sx={{
                    position: 'absolute',
                    insetInlineEnd: 0,
                    top: 0,
                    p: 1,
                  }}
                >
                  <Badge
                    variant="solid"
                    sx={
                      isSelectedForAutoAdd
                        ? AUTO_ADD_BADGE_STYLES
                        : BASE_BADGE_STYLES
                    }
                  >
                    {board.image_count}
                  </Badge>
                </Flex>
                <IAIDroppable
                  data={droppableData}
                  dropLabel={<Text fontSize="md">Move</Text>}
                />
              </Flex>

              <Flex
                sx={{
                  width: 'full',
                  height: 'full',
                  justifyContent: 'center',
                  alignItems: 'center',
                }}
              >
                <Editable
                  defaultValue={board_name}
                  submitOnBlur={true}
                  onSubmit={(nextValue) => {
                    handleUpdateBoardName(nextValue);
                  }}
                  sx={{ maxW: 'full' }}
                >
                  <EditablePreview
                    sx={{
                      color: isSelected
                        ? mode('base.900', 'base.50')(colorMode)
                        : mode('base.700', 'base.200')(colorMode),
                      fontWeight: isSelected ? 600 : undefined,
                      fontSize: 'xs',
                      textAlign: 'center',
                      p: 0,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                    noOfLines={1}
                  />
                  <EditableInput
                    sx={{
                      color: mode('base.900', 'base.50')(colorMode),
                      fontSize: 'xs',
                      borderColor: mode('base.500', 'base.500')(colorMode),
                      p: 0,
                      outline: 0,
                    }}
                  />
                </Editable>
              </Flex>
            </Flex>
          )}
        </BoardContextMenu>
      </Box>
    );
  }
);

GalleryBoard.displayName = 'HoverableBoard';

export default GalleryBoard;

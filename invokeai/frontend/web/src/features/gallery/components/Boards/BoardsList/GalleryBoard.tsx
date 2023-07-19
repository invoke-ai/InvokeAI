import {
  Badge,
  Box,
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  Image,
  MenuItem,
  MenuList,
  Text,
  useColorMode,
} from '@chakra-ui/react';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { MoveBoardDropData } from 'app/components/ImageDnd/typesafeDnd';
import { useAppDispatch } from 'app/store/storeHooks';
import { ContextMenu } from 'chakra-ui-contextmenu';
import IAIDroppable from 'common/components/IAIDroppable';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { FaTrash, FaUser } from 'react-icons/fa';
import { useUpdateBoardMutation } from 'services/api/endpoints/boards';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { BoardDTO } from 'services/api/types';
import { menuListMotionProps } from 'theme/components/menu';
import { mode } from 'theme/util/mode';

interface GalleryBoardProps {
  board: BoardDTO;
  isSelected: boolean;
  setBoardToDelete: (board?: BoardDTO) => void;
}

const GalleryBoard = memo(
  ({ board, isSelected, setBoardToDelete }: GalleryBoardProps) => {
    const dispatch = useAppDispatch();

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

    const handleDeleteBoard = useCallback(() => {
      setBoardToDelete(board);
    }, [board, setBoardToDelete]);

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
        <ContextMenu<HTMLDivElement>
          menuProps={{ size: 'sm', isLazy: true }}
          menuButtonProps={{
            bg: 'transparent',
            _hover: { bg: 'transparent' },
          }}
          renderMenu={() => (
            <MenuList
              sx={{ visibility: 'visible !important' }}
              motionProps={menuListMotionProps}
            >
              {board.image_count > 0 && (
                <>
                  {/* <MenuItem
                    isDisabled={!board.image_count}
                    icon={<FaImages />}
                    onClickCapture={handleAddBoardToBatch}
                  >
                    Add Board to Batch
                  </MenuItem> */}
                </>
              )}
              <MenuItem
                sx={{ color: 'error.600', _dark: { color: 'error.300' } }}
                icon={<FaTrash />}
                onClickCapture={handleDeleteBoard}
              >
                Delete Board
              </MenuItem>
            </MenuList>
          )}
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
                  <Badge variant="solid">{board.image_count}</Badge>
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
                  submitOnBlur={false}
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
        </ContextMenu>
      </Box>
    );
  }
);

GalleryBoard.displayName = 'HoverableBoard';

export default GalleryBoard;

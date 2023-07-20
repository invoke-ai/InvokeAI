import {
  Badge,
  Box,
  ChakraProps,
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  Icon,
  Image,
  Text,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { MoveBoardDropData } from 'app/components/ImageDnd/typesafeDnd';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDroppable from 'common/components/IAIDroppable';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo, useState } from 'react';
import { FaFolder } from 'react-icons/fa';
import { useUpdateBoardMutation } from 'services/api/endpoints/boards';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { BoardDTO } from 'services/api/types';
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

    const { board_name, board_id } = board;
    const [localBoardName, setLocalBoardName] = useState(board_name);

    const handleSelectBoard = useCallback(() => {
      dispatch(boardIdSelected(board_id));
    }, [board_id, dispatch]);

    const [updateBoard, { isLoading: isUpdateBoardLoading }] =
      useUpdateBoardMutation();

    const droppableData: MoveBoardDropData = useMemo(
      () => ({
        id: board_id,
        actionType: 'MOVE_BOARD',
        context: { boardId: board_id },
      }),
      [board_id]
    );

    const handleSubmit = useCallback(
      (newBoardName: string) => {
        if (!newBoardName) {
          // empty strings are not allowed
          setLocalBoardName(board_name);
          return;
        }
        if (newBoardName === board_name) {
          // don't updated the board name if it hasn't changed
          return;
        }
        updateBoard({ board_id, changes: { board_name: newBoardName } })
          .unwrap()
          .then((response) => {
            // update local state
            setLocalBoardName(response.board_name);
          })
          .catch(() => {
            // revert on error
            setLocalBoardName(board_name);
          });
      },
      [board_id, board_name, updateBoard]
    );

    const handleChange = useCallback((newBoardName: string) => {
      setLocalBoardName(newBoardName);
    }, []);

    return (
      <Box
        sx={{ w: 'full', h: 'full', touchAction: 'none', userSelect: 'none' }}
      >
        <Flex
          sx={{
            position: 'relative',
            justifyContent: 'center',
            alignItems: 'center',
            aspectRatio: '1/1',
            w: 'full',
            h: 'full',
          }}
        >
          <BoardContextMenu
            board={board}
            board_id={board_id}
            setBoardToDelete={setBoardToDelete}
          >
            {(ref) => (
              <Flex
                ref={ref}
                onClick={handleSelectBoard}
                sx={{
                  w: 'full',
                  h: 'full',
                  position: 'relative',
                  justifyContent: 'center',
                  alignItems: 'center',
                  borderRadius: 'base',
                  cursor: 'pointer',
                }}
              >
                <Flex
                  sx={{
                    w: 'full',
                    h: 'full',
                    justifyContent: 'center',
                    alignItems: 'center',
                    borderRadius: 'base',
                    bg: 'base.200',
                    _dark: {
                      bg: 'base.800',
                    },
                  }}
                >
                  {coverImage?.thumbnail_url ? (
                    <Image
                      src={coverImage?.thumbnail_url}
                      draggable={false}
                      sx={{
                        maxW: 'full',
                        maxH: 'full',
                        borderRadius: 'base',
                        borderBottomRadius: 'lg',
                      }}
                    />
                  ) : (
                    <Flex
                      sx={{
                        w: 'full',
                        h: 'full',
                        justifyContent: 'center',
                        alignItems: 'center',
                      }}
                    >
                      <Icon
                        boxSize={12}
                        as={FaFolder}
                        sx={{
                          mt: -3,
                          opacity: 0.7,
                          color: 'base.500',
                          _dark: {
                            color: 'base.500',
                          },
                        }}
                      />
                    </Flex>
                  )}
                </Flex>
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
                <Box
                  className="selection-box"
                  sx={{
                    position: 'absolute',
                    top: 0,
                    insetInlineEnd: 0,
                    bottom: 0,
                    insetInlineStart: 0,
                    borderRadius: 'base',
                    transitionProperty: 'common',
                    transitionDuration: 'common',
                    shadow: isSelected ? 'selected.light' : undefined,
                    _dark: {
                      shadow: isSelected ? 'selected.dark' : undefined,
                    },
                  }}
                />
                <Flex
                  sx={{
                    position: 'absolute',
                    bottom: 0,
                    left: 0,
                    p: 1,
                    justifyContent: 'center',
                    alignItems: 'center',
                    w: 'full',
                    maxW: 'full',
                    borderBottomRadius: 'base',
                    bg: 'accent.400',
                    color: isSelected ? 'base.50' : 'base.100',
                    _dark: { color: 'base.200', bg: 'accent.500' },
                    lineHeight: 'short',
                    fontSize: 'xs',
                  }}
                >
                  <Editable
                    value={localBoardName}
                    isDisabled={isUpdateBoardLoading}
                    submitOnBlur={true}
                    onChange={handleChange}
                    onSubmit={handleSubmit}
                    sx={{
                      w: 'full',
                    }}
                  >
                    <EditablePreview
                      sx={{
                        p: 0,
                        fontWeight: isSelected ? 700 : 500,
                        textAlign: 'center',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      }}
                      noOfLines={1}
                    />
                    <EditableInput
                      sx={{
                        p: 0,
                        _focusVisible: {
                          p: 0,
                          // get rid of the edit border
                          boxShadow: 'none',
                        },
                      }}
                    />
                  </Editable>
                </Flex>

                <IAIDroppable
                  data={droppableData}
                  dropLabel={<Text fontSize="md">Move</Text>}
                />
              </Flex>
            )}
          </BoardContextMenu>
        </Flex>
      </Box>
    );
  }
);

GalleryBoard.displayName = 'HoverableBoard';

export default GalleryBoard;

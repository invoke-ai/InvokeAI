import {
  Box,
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  Icon,
  Image,
  Text,
  Tooltip,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDroppable from 'common/components/IAIDroppable';
import SelectionOverlay from 'common/components/SelectionOverlay';
import {
  autoAddBoardIdChanged,
  boardIdSelected,
} from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo, useState } from 'react';
import { FaUser } from 'react-icons/fa';
import {
  useGetBoardAssetsTotalQuery,
  useGetBoardImagesTotalQuery,
  useUpdateBoardMutation,
} from 'services/api/endpoints/boards';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { BoardDTO } from 'services/api/types';
import AutoAddIcon from '../AutoAddIcon';
import BoardContextMenu from '../BoardContextMenu';
import { AddToBoardDropData } from 'features/dnd/types';

interface GalleryBoardProps {
  board: BoardDTO;
  isSelected: boolean;
  setBoardToDelete: (board?: BoardDTO) => void;
}

const GalleryBoard = ({
  board,
  isSelected,
  setBoardToDelete,
}: GalleryBoardProps) => {
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ gallery, system }) => {
          const isSelectedForAutoAdd =
            board.board_id === gallery.autoAddBoardId;
          const autoAssignBoardOnClick = gallery.autoAssignBoardOnClick;
          const isProcessing = system.isProcessing;

          return {
            isSelectedForAutoAdd,
            autoAssignBoardOnClick,
            isProcessing,
          };
        },
        defaultSelectorOptions
      ),
    [board.board_id]
  );

  const { isSelectedForAutoAdd, autoAssignBoardOnClick, isProcessing } =
    useAppSelector(selector);
  const [isHovered, setIsHovered] = useState(false);
  const handleMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);
  const handleMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const { data: imagesTotal } = useGetBoardImagesTotalQuery(board.board_id);
  const { data: assetsTotal } = useGetBoardAssetsTotalQuery(board.board_id);
  const tooltip = useMemo(() => {
    if (!imagesTotal || !assetsTotal) {
      return undefined;
    }
    return `${imagesTotal} image${
      imagesTotal > 1 ? 's' : ''
    }, ${assetsTotal} asset${assetsTotal > 1 ? 's' : ''}`;
  }, [assetsTotal, imagesTotal]);

  const { currentData: coverImage } = useGetImageDTOQuery(
    board.cover_image_name ?? skipToken
  );

  const { board_name, board_id } = board;
  const [localBoardName, setLocalBoardName] = useState(board_name);

  const handleSelectBoard = useCallback(() => {
    dispatch(boardIdSelected(board_id));
    if (autoAssignBoardOnClick && !isProcessing) {
      dispatch(autoAddBoardIdChanged(board_id));
    }
  }, [board_id, autoAssignBoardOnClick, isProcessing, dispatch]);

  const [updateBoard, { isLoading: isUpdateBoardLoading }] =
    useUpdateBoardMutation();

  const droppableData: AddToBoardDropData = useMemo(
    () => ({
      id: board_id,
      actionType: 'ADD_TO_BOARD',
      context: { boardId: board_id },
    }),
    [board_id]
  );

  const handleSubmit = useCallback(
    async (newBoardName: string) => {
      // empty strings are not allowed
      if (!newBoardName.trim()) {
        setLocalBoardName(board_name);
        return;
      }

      // don't updated the board name if it hasn't changed
      if (newBoardName === board_name) {
        return;
      }

      try {
        const { board_name } = await updateBoard({
          board_id,
          changes: { board_name: newBoardName },
        }).unwrap();

        // update local state
        setLocalBoardName(board_name);
      } catch {
        // revert on error
        setLocalBoardName(board_name);
      }
    },
    [board_id, board_name, updateBoard]
  );

  const handleChange = useCallback((newBoardName: string) => {
    setLocalBoardName(newBoardName);
  }, []);

  return (
    <Box sx={{ w: 'full', h: 'full', touchAction: 'none', userSelect: 'none' }}>
      <Flex
        onMouseOver={handleMouseOver}
        onMouseOut={handleMouseOut}
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
            <Tooltip label={tooltip} openDelay={1000} hasArrow>
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
                      objectFit: 'cover',
                      w: 'full',
                      h: 'full',
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
                      as={FaUser}
                      sx={{
                        mt: -6,
                        opacity: 0.7,
                        color: 'base.500',
                        _dark: {
                          color: 'base.500',
                        },
                      }}
                    />
                  </Flex>
                )}
                {/* <Flex
                  sx={{
                    position: 'absolute',
                    insetInlineEnd: 0,
                    top: 0,
                    p: 1,
                  }}
                >
                  <Badge variant="solid" sx={BASE_BADGE_STYLES}>
                    {totalImages}/{totalAssets}
                  </Badge>
                </Flex> */}
                {isSelectedForAutoAdd && <AutoAddIcon />}
                <SelectionOverlay
                  isSelected={isSelected}
                  isHovered={isHovered}
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
                    bg: isSelected ? 'accent.400' : 'base.500',
                    color: isSelected ? 'base.50' : 'base.100',
                    _dark: {
                      bg: isSelected ? 'accent.500' : 'base.600',
                      color: isSelected ? 'base.50' : 'base.100',
                    },
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
                          textAlign: 'center',
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
            </Tooltip>
          )}
        </BoardContextMenu>
      </Flex>
    </Box>
  );
};

export default memo(GalleryBoard);

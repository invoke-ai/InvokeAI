import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  Icon,
  Image,
  Text,
  Tooltip,
  useDisclosure,
  useEditableControls,
} from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import type { AddToBoardDropData } from 'features/dnd/types';
import { AutoAddBadge } from 'features/gallery/components/Boards/AutoAddBadge';
import BoardContextMenu from 'features/gallery/components/Boards/BoardContextMenu';
import { BoardTooltip } from 'features/gallery/components/Boards/BoardsList/BoardTooltip';
import {
  selectAutoAddBoardId,
  selectAutoAssignBoardOnClick,
  selectSelectedBoardId,
} from 'features/gallery/store/gallerySelectors';
import { autoAddBoardIdChanged, boardIdSelected } from 'features/gallery/store/gallerySlice';
import type { MouseEvent, MouseEventHandler, MutableRefObject } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArchiveBold, PiImageSquare } from 'react-icons/pi';
import { useUpdateBoardMutation } from 'services/api/endpoints/boards';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { BoardDTO } from 'services/api/types';

const editableInputStyles: SystemStyleObject = {
  p: 0,
  fontSize: 'md',
  w: '100%',
  _focusVisible: {
    p: 0,
  },
};

const _hover: SystemStyleObject = {
  bg: 'base.850',
};

interface GalleryBoardProps {
  board: BoardDTO;
  isSelected: boolean;
}

const GalleryBoard = ({ board, isSelected }: GalleryBoardProps) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const autoAssignBoardOnClick = useAppSelector(selectAutoAssignBoardOnClick);
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const editingDisclosure = useDisclosure();
  const [localBoardName, setLocalBoardName] = useState(board.board_name);
  const onStartEditingRef = useRef<MouseEventHandler | undefined>(undefined);

  const onPointerUp = useCallback(() => {
    if (selectedBoardId !== board.board_id) {
      dispatch(boardIdSelected({ boardId: board.board_id }));
    }
    if (autoAssignBoardOnClick && autoAddBoardId !== board.board_id) {
      dispatch(autoAddBoardIdChanged(board.board_id));
    }
  }, [selectedBoardId, board.board_id, autoAssignBoardOnClick, autoAddBoardId, dispatch]);

  const [updateBoard, { isLoading: isUpdateBoardLoading }] = useUpdateBoardMutation();

  const droppableData: AddToBoardDropData = useMemo(
    () => ({
      id: board.board_id,
      actionType: 'ADD_TO_BOARD',
      context: { boardId: board.board_id },
    }),
    [board.board_id]
  );

  const onSubmit = useCallback(
    async (newBoardName: string) => {
      if (!newBoardName.trim()) {
        // empty strings are not allowed
        setLocalBoardName(board.board_name);
      } else if (newBoardName === board.board_name) {
        // don't updated the board name if it hasn't changed
      } else {
        try {
          const { board_name } = await updateBoard({
            board_id: board.board_id,
            changes: { board_name: newBoardName },
          }).unwrap();

          // update local state
          setLocalBoardName(board_name);
        } catch {
          // revert on error
          setLocalBoardName(board.board_name);
        }
      }
      editingDisclosure.onClose();
    },
    [board.board_id, board.board_name, editingDisclosure, updateBoard]
  );

  const onChange = useCallback((newBoardName: string) => {
    setLocalBoardName(newBoardName);
  }, []);

  const onDoubleClick = useCallback((e: MouseEvent<HTMLDivElement>) => {
    if (onStartEditingRef.current) {
      onStartEditingRef.current(e);
    }
  }, []);

  return (
    <BoardContextMenu board={board}>
      {(ref) => (
        <Tooltip label={<BoardTooltip board={board} />} openDelay={1000} placement="left" closeOnScroll p={2}>
          <Flex
            position="relative"
            ref={ref}
            onPointerUp={onPointerUp}
            onDoubleClick={onDoubleClick}
            w="full"
            alignItems="center"
            borderRadius="base"
            cursor="pointer"
            py={1}
            ps={1}
            pe={4}
            gap={4}
            bg={isSelected ? 'base.850' : undefined}
            _hover={_hover}
            h={12}
          >
            <CoverImage board={board} />
            <Editable
              as={Flex}
              alignItems="center"
              gap={4}
              flexGrow={1}
              onEdit={editingDisclosure.onOpen}
              value={localBoardName}
              isDisabled={isUpdateBoardLoading}
              submitOnBlur={true}
              onChange={onChange}
              onSubmit={onSubmit}
              isPreviewFocusable={false}
              fontSize="sm"
            >
              <EditablePreview
                cursor="pointer"
                p={0}
                fontSize="sm"
                textOverflow="ellipsis"
                noOfLines={1}
                w="fit-content"
                wordBreak="break-all"
                fontWeight={isSelected ? 'bold' : 'normal'}
              />
              <EditableInput sx={editableInputStyles} />
              <JankEditableHijack onStartEditingRef={onStartEditingRef} />
            </Editable>
            {autoAddBoardId === board.board_id && !editingDisclosure.isOpen && <AutoAddBadge />}
            {board.archived && !editingDisclosure.isOpen && <Icon as={PiArchiveBold} fill="base.300" />}
            {!editingDisclosure.isOpen && <Text variant="subtext">{board.image_count}</Text>}

            <IAIDroppable data={droppableData} dropLabel={t('gallery.move')} />
          </Flex>
        </Tooltip>
      )}
    </BoardContextMenu>
  );
};

const JankEditableHijack = memo((props: { onStartEditingRef: MutableRefObject<MouseEventHandler | undefined> }) => {
  const editableControls = useEditableControls();
  useEffect(() => {
    props.onStartEditingRef.current = editableControls.getEditButtonProps().onPointerUp;
  }, [props, editableControls]);
  return null;
});

JankEditableHijack.displayName = 'JankEditableHijack';

export default memo(GalleryBoard);

const CoverImage = ({ board }: { board: BoardDTO }) => {
  const { currentData: coverImage } = useGetImageDTOQuery(board.cover_image_name ?? skipToken);

  if (coverImage) {
    return (
      <Image
        src={coverImage.thumbnail_url}
        draggable={false}
        objectFit="cover"
        w={10}
        h={10}
        borderRadius="base"
        borderBottomRadius="lg"
      />
    );
  }

  return (
    <Flex w={10} h={10} justifyContent="center" alignItems="center">
      <Icon boxSize={10} as={PiImageSquare} opacity={0.7} color="base.500" />
    </Flex>
  );
};

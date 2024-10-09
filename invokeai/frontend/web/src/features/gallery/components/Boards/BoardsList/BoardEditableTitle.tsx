import { Flex, IconButton, Input, Text } from '@invoke-ai/ui-library';
import { useBoolean } from 'common/hooks/useBoolean';
import { withResultAsync } from 'common/util/result';
import { toast } from 'features/toast/toast';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPencilBold } from 'react-icons/pi';
import { useUpdateBoardMutation } from 'services/api/endpoints/boards';
import type { BoardDTO } from 'services/api/types';

type Props = {
  board: BoardDTO;
  isSelected: boolean;
};

export const BoardEditableTitle = memo(({ board, isSelected }: Props) => {
  const { t } = useTranslation();
  const isEditing = useBoolean(false);
  const [isHovering, setIsHovering] = useState(false);
  const [localTitle, setLocalTitle] = useState(board.board_name);
  const ref = useRef<HTMLInputElement>(null);
  const [updateBoard, updateBoardResult] = useUpdateBoardMutation();

  const onChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setLocalTitle(e.target.value);
  }, []);

  const onEdit = useCallback(() => {
    isEditing.setTrue();
    setIsHovering(false);
  }, [isEditing]);

  const onBlur = useCallback(async () => {
    const trimmedTitle = localTitle.trim();
    isEditing.setFalse();
    if (trimmedTitle.length === 0) {
      setLocalTitle(board.board_name);
    } else if (trimmedTitle !== board.board_name) {
      setLocalTitle(trimmedTitle);
      const result = await withResultAsync(() =>
        updateBoard({ board_id: board.board_id, changes: { board_name: trimmedTitle } }).unwrap()
      );
      if (result.isErr()) {
        setLocalTitle(board.board_name);
        toast({
          status: 'error',
          title: t('boards.updateBoardError'),
        });
      } else {
        setLocalTitle(result.value.board_name);
      }
    }
  }, [board.board_id, board.board_name, isEditing, localTitle, updateBoard, t]);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        onBlur();
      } else if (e.key === 'Escape') {
        setLocalTitle(board.board_name);
        isEditing.setFalse();
      }
    },
    [board.board_name, isEditing, onBlur]
  );

  const handleMouseOver = useCallback(() => {
    setIsHovering(true);
  }, []);

  const handleMouseOut = useCallback(() => {
    setIsHovering(false);
  }, []);

  useEffect(() => {
    if (isEditing.isTrue) {
      ref.current?.focus();
      ref.current?.select();
    }
  }, [isEditing.isTrue]);

  if (!isEditing.isTrue) {
    return (
      <Flex alignItems="center" gap={3} onMouseOver={handleMouseOver} onMouseOut={handleMouseOut}>
        <Text
          size="sm"
          fontWeight="semibold"
          userSelect="none"
          color={isSelected ? 'base.100' : 'base.300'}
          onDoubleClick={onEdit}
          cursor="text"
        >
          {localTitle}
        </Text>
        {isHovering && (
          <IconButton aria-label="edit name" icon={<PiPencilBold />} size="sm" variant="ghost" onClick={onEdit} />
        )}
      </Flex>
    );
  }

  return (
    <Input
      ref={ref}
      value={localTitle}
      onChange={onChange}
      onBlur={onBlur}
      onKeyDown={onKeyDown}
      variant="outline"
      isDisabled={updateBoardResult.isLoading}
      _focusVisible={{ borderWidth: 1, borderColor: 'invokeBlueAlpha.400', borderRadius: 'base' }}
    />
  );
});

BoardEditableTitle.displayName = 'CanvasEntityTitleEdit';

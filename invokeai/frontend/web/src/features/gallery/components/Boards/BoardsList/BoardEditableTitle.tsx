import { Input, Text } from '@invoke-ai/ui-library';
import { useBoolean } from 'common/hooks/useBoolean';
import { withResultAsync } from 'common/util/result';
import { toast } from 'features/toast/toast';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useUpdateBoardMutation } from 'services/api/endpoints/boards';
import type { BoardDTO } from 'services/api/types';

type Props = {
  board: BoardDTO;
  isSelected: boolean;
};

export const BoardEditableTitle = memo(({ board, isSelected }: Props) => {
  const { t } = useTranslation();
  const isEditing = useBoolean(false);
  const [localTitle, setLocalTitle] = useState(board.board_name);
  const ref = useRef<HTMLInputElement>(null);
  const [updateBoard, updateBoardResult] = useUpdateBoardMutation();

  const onChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setLocalTitle(e.target.value);
  }, []);

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

  useEffect(() => {
    if (isEditing.isTrue) {
      ref.current?.focus();
      ref.current?.select();
    }
  }, [isEditing.isTrue]);

  if (!isEditing.isTrue) {
    return (
      <Text
        size="sm"
        fontWeight="semibold"
        userSelect="none"
        color={isSelected ? 'base.100' : 'base.300'}
        onDoubleClick={isEditing.setTrue}
        cursor="text"
        minW={16}
      >
        {localTitle}
      </Text>
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

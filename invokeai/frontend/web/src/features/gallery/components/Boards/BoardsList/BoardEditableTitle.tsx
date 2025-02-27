import { Flex, IconButton, Input, Text } from '@invoke-ai/ui-library';
import { useBoolean } from 'common/hooks/useBoolean';
import { useEditable } from 'common/hooks/useEditable';
import { withResultAsync } from 'common/util/result';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useRef } from 'react';
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
  const isHovering = useBoolean(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const [updateBoard, updateBoardResult] = useUpdateBoardMutation();

  const onChange = useCallback(
    async (board_name: string) => {
      const result = await withResultAsync(() =>
        updateBoard({ board_id: board.board_id, changes: { board_name } }).unwrap()
      );
      if (result.isErr()) {
        toast({
          status: 'error',
          title: t('boards.updateBoardError'),
        });
      }
    },
    [board.board_id, t, updateBoard]
  );

  const editable = useEditable({
    value: board.board_name,
    defaultValue: board.board_name,
    onChange,
    inputRef,
    onStartEditing: isHovering.setTrue,
  });

  if (!editable.isEditing) {
    return (
      <Flex alignItems="center" gap={3} onMouseOver={isHovering.setTrue} onMouseOut={isHovering.setFalse}>
        <Text
          size="sm"
          fontWeight="semibold"
          userSelect="none"
          color={isSelected ? 'base.100' : 'base.300'}
          onDoubleClick={editable.startEditing}
          cursor="text"
        >
          {editable.value}
        </Text>
        {isHovering.isTrue && (
          <IconButton
            aria-label="edit name"
            icon={<PiPencilBold />}
            size="sm"
            variant="ghost"
            onClick={editable.startEditing}
          />
        )}
      </Flex>
    );
  }

  return (
    <Input
      ref={inputRef}
      {...editable.inputProps}
      variant="outline"
      isDisabled={updateBoardResult.isLoading}
      _focusVisible={{ borderWidth: 1, borderColor: 'invokeBlueAlpha.400', borderRadius: 'base' }}
    />
  );
});

BoardEditableTitle.displayName = 'CanvasEntityTitleEdit';

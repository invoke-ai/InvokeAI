import { Flex, IconButton, Input, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useBoolean } from 'common/hooks/useBoolean';
import { useEditable } from 'common/hooks/useEditable';
import { canvasNameChanged } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback, useRef } from 'react';
import { PiPencilBold } from 'react-icons/pi';

interface CanvasTabEditableTitleProps {
  name: string;
  isActive: boolean;
}

export const CanvasTabEditableTitle = memo(({ name, isActive }: CanvasTabEditableTitleProps) => {
  const dispatch = useAppDispatch();
  const isHovering = useBoolean(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const onChange = useCallback(
    (value: string) => {
      dispatch(canvasNameChanged({ name: value }));
    },
    [dispatch]
  );

  const editable = useEditable({
    value: name,
    defaultValue: name,
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
          color={isActive ? 'base.100' : 'base.300'}
          onDoubleClick={editable.startEditing}
          cursor="text"
          noOfLines={1}
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
      textAlign="center"
      _focusVisible={{ borderWidth: 1, borderColor: 'invokeBlueAlpha.400', borderRadius: 'base' }}
    />
  );
});
CanvasTabEditableTitle.displayName = 'CanvasTabEditableTitle';

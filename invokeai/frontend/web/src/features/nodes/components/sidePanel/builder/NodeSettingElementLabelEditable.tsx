import { Flex, FormLabel, Input } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { useNodeSettingDefaultLabel } from 'features/nodes/hooks/useNodeSetting';
import { formElementNodeSettingDataChanged } from 'features/nodes/store/nodesSlice';
import type { NodeSettingElement } from 'features/nodes/types/workflow';
import { memo, useCallback, useRef } from 'react';

export const NodeSettingElementLabelEditable = memo(({ el }: { el: NodeSettingElement }) => {
  const { id, data } = el;
  const dispatch = useAppDispatch();
  const defaultLabel = useNodeSettingDefaultLabel(data.setting);
  const inputRef = useRef<HTMLInputElement>(null);

  const onChange = useCallback(
    (label: string) => {
      dispatch(formElementNodeSettingDataChanged({ id, changes: { label } }));
    },
    [dispatch, id]
  );

  const editable = useEditable({
    value: data.label || defaultLabel,
    defaultValue: defaultLabel,
    inputRef,
    onChange,
  });

  if (!editable.isEditing) {
    return (
      <Flex w="full" gap={4}>
        <FormLabel onDoubleClick={editable.startEditing} cursor="text">
          {editable.value}
        </FormLabel>
      </Flex>
    );
  }

  return (
    <Input
      ref={inputRef}
      variant="outline"
      p={1}
      px={2}
      _focusVisible={{ borderRadius: 'base', h: 'unset' }}
      {...editable.inputProps}
    />
  );
});
NodeSettingElementLabelEditable.displayName = 'NodeSettingElementLabelEditable';

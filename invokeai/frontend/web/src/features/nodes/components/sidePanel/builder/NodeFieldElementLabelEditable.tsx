import { Flex, FormLabel, Input, Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { NodeFieldElementResetToInitialValueIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/NodeFieldElementResetToInitialValueIconButton';
import { useInputFieldTemplateOrThrow } from 'features/nodes/hooks/useInputFieldTemplateOrThrow';
import { useInputFieldUserTitleSafe } from 'features/nodes/hooks/useInputFieldUserTitleSafe';
import { fieldLabelChanged } from 'features/nodes/store/nodesSlice';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { memo, useCallback, useRef } from 'react';

export const NodeFieldElementLabelEditable = memo(({ el }: { el: NodeFieldElement }) => {
  const { data } = el;
  const { fieldIdentifier } = data;
  const dispatch = useAppDispatch();
  const label = useInputFieldUserTitleSafe(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const fieldTemplate = useInputFieldTemplateOrThrow(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const inputRef = useRef<HTMLInputElement>(null);

  const onChange = useCallback(
    (label: string) => {
      dispatch(fieldLabelChanged({ nodeId: fieldIdentifier.nodeId, fieldName: fieldIdentifier.fieldName, label }));
    },
    [dispatch, fieldIdentifier.fieldName, fieldIdentifier.nodeId]
  );

  const editable = useEditable({
    value: label || fieldTemplate.title,
    defaultValue: fieldTemplate.title,
    inputRef,
    onChange,
  });

  if (!editable.isEditing) {
    return (
      <Flex w="full" gap={4}>
        <FormLabel onDoubleClick={editable.startEditing} cursor="text">
          {editable.value}
        </FormLabel>
        <Spacer />
        <NodeFieldElementResetToInitialValueIconButton element={el} />
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
NodeFieldElementLabelEditable.displayName = 'NodeFieldElementLabelEditable';

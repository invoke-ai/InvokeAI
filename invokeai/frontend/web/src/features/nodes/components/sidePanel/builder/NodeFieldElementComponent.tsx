import { Flex } from '@invoke-ai/ui-library';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { InputFieldViewMode } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldViewMode';
import { useElement } from 'features/nodes/store/workflowSlice';
import { isNodeFieldElement, NODE_FIELD_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const NodeFieldElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);

  if (!el || !isNodeFieldElement(el)) {
    return null;
  }

  const { fieldIdentifier } = el.data;

  return (
    <Flex id={id} className={NODE_FIELD_CLASS_NAME}>
      <InputFieldGate nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName}>
        <InputFieldViewMode nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName} />
      </InputFieldGate>
    </Flex>
  );
});

NodeFieldElementComponent.displayName = 'NodeFieldElementComponent';

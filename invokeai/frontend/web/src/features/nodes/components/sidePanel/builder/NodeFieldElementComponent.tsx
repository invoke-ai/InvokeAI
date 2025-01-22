import { Flex } from '@invoke-ai/ui-library';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { InputFieldViewMode } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldViewMode';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const NodeFieldElementComponent = memo(({ element }: { element: NodeFieldElement }) => {
  const { id, data } = element;
  const { fieldIdentifier } = data;

  return (
    <Flex id={id} flexBasis="100%">
      <InputFieldGate nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName}>
        <InputFieldViewMode nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName} />
      </InputFieldGate>
    </Flex>
  );
});

NodeFieldElementComponent.displayName = 'NodeFieldElementComponent';

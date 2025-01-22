import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { InputFieldViewMode } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldViewMode';
import { ElementWrapper } from 'features/nodes/components/sidePanel/builder/ElementWrapper';
import { useElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const NodeFieldElementComponent = memo(({ id }: { id: string }) => {
  const element = useElement(id);

  if (!element || element.type !== 'node-field') {
    return null;
  }
  const { data } = element;
  const { fieldIdentifier } = data;

  return (
    <ElementWrapper id={id}>
      <InputFieldGate nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName}>
        <InputFieldViewMode nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName} />
      </InputFieldGate>
    </ElementWrapper>
  );
});

NodeFieldElementComponent.displayName = 'NodeFieldElementComponent';

import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { InputFieldViewMode } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldViewMode';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { selectWorkflowFormMode, useElement } from 'features/nodes/store/workflowSlice';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { isNodeFieldElement, NODE_FIELD_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const NodeFieldElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowFormMode);

  if (!el || !isNodeFieldElement(el)) {
    return null;
  }

  if (mode === 'view') {
    return <NodeFieldElementComponentViewMode el={el} />;
  }

  // mode === 'edit'
  return <NodeFieldElementComponentEditMode el={el} />;
});

NodeFieldElementComponent.displayName = 'NodeFieldElementComponent';

export const NodeFieldElementComponentViewMode = memo(({ el }: { el: NodeFieldElement }) => {
  const { id, data } = el;
  const { fieldIdentifier } = data;

  return (
    <Flex id={id} className={NODE_FIELD_CLASS_NAME}>
      <InputFieldGate nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName}>
        <InputFieldViewMode nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName} />
      </InputFieldGate>
    </Flex>
  );
});

NodeFieldElementComponentViewMode.displayName = 'NodeFieldElementComponentViewMode';

export const NodeFieldElementComponentEditMode = memo(({ el }: { el: NodeFieldElement }) => {
  const { id, data } = el;
  const { fieldIdentifier } = data;

  return (
    <FormElementEditModeWrapper element={el}>
      <Flex id={id} className={NODE_FIELD_CLASS_NAME} w="full">
        <InputFieldGate nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName}>
          <InputFieldViewMode nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName} />
        </InputFieldGate>
      </Flex>
    </FormElementEditModeWrapper>
  );
});

NodeFieldElementComponentEditMode.displayName = 'NodeFieldElementComponentEditMode';

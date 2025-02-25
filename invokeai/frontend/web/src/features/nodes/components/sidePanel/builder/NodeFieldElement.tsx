import { useAppSelector } from 'app/store/storeHooks';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { NodeFieldElementEditMode } from 'features/nodes/components/sidePanel/builder/NodeFieldElementEditMode';
import { NodeFieldElementViewMode } from 'features/nodes/components/sidePanel/builder/NodeFieldElementViewMode';
import { selectWorkflowMode, useElement } from 'features/nodes/store/workflowSlice';
import { isNodeFieldElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const NodeFieldElement = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowMode);

  if (!el || !isNodeFieldElement(el)) {
    return null;
  }

  if (mode === 'view') {
    return (
      <InputFieldGate nodeId={el.data.fieldIdentifier.nodeId} fieldName={el.data.fieldIdentifier.fieldName}>
        <NodeFieldElementViewMode el={el} />
      </InputFieldGate>
    );
  }

  // mode === 'edit'
  return (
    <InputFieldGate nodeId={el.data.fieldIdentifier.nodeId} fieldName={el.data.fieldIdentifier.fieldName}>
      <NodeFieldElementEditMode el={el} />
    </InputFieldGate>
  );
});

NodeFieldElement.displayName = 'NodeFieldElement';

import { useAppSelector } from 'app/store/storeHooks';
import { InvocationNodeContextProvider } from 'features/nodes/components/flow/nodes/Invocation/context';
import { NodeFieldElementEditMode } from 'features/nodes/components/sidePanel/builder/NodeFieldElementEditMode';
import { NodeFieldElementViewMode } from 'features/nodes/components/sidePanel/builder/NodeFieldElementViewMode';
import { useElement } from 'features/nodes/components/sidePanel/builder/use-element';
import { selectWorkflowMode } from 'features/nodes/store/workflowLibrarySlice';
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
      <InvocationNodeContextProvider nodeId={el.data.fieldIdentifier.nodeId}>
        <NodeFieldElementViewMode el={el} />
      </InvocationNodeContextProvider>
    );
  }

  // mode === 'edit'
  return (
    <InvocationNodeContextProvider nodeId={el.data.fieldIdentifier.nodeId}>
      <NodeFieldElementEditMode el={el} />
    </InvocationNodeContextProvider>
  );
});

NodeFieldElement.displayName = 'NodeFieldElement';

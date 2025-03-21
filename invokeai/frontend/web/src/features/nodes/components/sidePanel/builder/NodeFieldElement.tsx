import { useAppSelector } from 'app/store/storeHooks';
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
    return <NodeFieldElementViewMode el={el} />;
  }

  // mode === 'edit'
  return <NodeFieldElementEditMode el={el} />;
});

NodeFieldElement.displayName = 'NodeFieldElement';

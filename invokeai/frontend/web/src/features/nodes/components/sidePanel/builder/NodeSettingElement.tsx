import { useAppSelector } from 'app/store/storeHooks';
import { InvocationNodeContextProvider } from 'features/nodes/components/flow/nodes/Invocation/context';
import { NodeSettingElementEditMode } from 'features/nodes/components/sidePanel/builder/NodeSettingElementEditMode';
import { NodeSettingElementViewMode } from 'features/nodes/components/sidePanel/builder/NodeSettingElementViewMode';
import { useElement } from 'features/nodes/components/sidePanel/builder/use-element';
import { selectWorkflowMode } from 'features/nodes/store/workflowLibrarySlice';
import { isNodeSettingElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const NodeSettingElement = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowMode);

  if (!el || !isNodeSettingElement(el)) {
    return null;
  }

  if (mode === 'view') {
    return (
      <InvocationNodeContextProvider nodeId={el.data.nodeId}>
        <NodeSettingElementViewMode el={el} />
      </InvocationNodeContextProvider>
    );
  }

  // mode === 'edit'
  return (
    <InvocationNodeContextProvider nodeId={el.data.nodeId}>
      <NodeSettingElementEditMode el={el} />
    </InvocationNodeContextProvider>
  );
});

NodeSettingElement.displayName = 'NodeSettingElement';

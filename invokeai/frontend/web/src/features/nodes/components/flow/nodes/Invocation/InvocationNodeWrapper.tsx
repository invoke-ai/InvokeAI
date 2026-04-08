import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import type { Node, NodeProps } from '@xyflow/react';
import { useAppSelector } from 'app/store/storeHooks';
import NodeWrapper from 'features/nodes/components/flow/nodes/common/NodeWrapper';
import InvocationNode from 'features/nodes/components/flow/nodes/Invocation/InvocationNode';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodes } from 'features/nodes/store/selectors';
import type { InvocationNodeData } from 'features/nodes/types/invocation';
import { memo, useMemo } from 'react';

import CallSavedWorkflowsNode from './CallSavedWorkflowsNode';
import { InvocationNodeContextProvider } from './context';
import { getInvocationNodeBodyComponentKey } from './getInvocationNodeBodyComponent';
import InvocationNodeUnknownFallback from './InvocationNodeUnknownFallback';

const InvocationNodeWrapper = (props: NodeProps<Node<InvocationNodeData>>) => {
  const { data, selected } = props;
  const { id: nodeId, type, isOpen, label } = data;
  const templates = useStore($templates);
  const hasTemplate = useMemo(() => Boolean(templates[type]), [templates, type]);
  const bodyComponentKey = useMemo(() => getInvocationNodeBodyComponentKey(type), [type]);
  const selectNodeExists = useMemo(
    () => createSelector(selectNodes, (nodes) => Boolean(nodes.find((n) => n.id === nodeId))),
    [nodeId]
  );
  const nodeExists = useAppSelector(selectNodeExists);

  if (!nodeExists) {
    return null;
  }

  if (!hasTemplate) {
    return (
      <InvocationNodeContextProvider nodeId={nodeId}>
        <NodeWrapper nodeId={nodeId} selected={selected} isMissingTemplate>
          <InvocationNodeUnknownFallback nodeId={nodeId} isOpen={isOpen} label={label} type={type} />
        </NodeWrapper>
      </InvocationNodeContextProvider>
    );
  }

  return (
    <InvocationNodeContextProvider nodeId={nodeId}>
      <NodeWrapper nodeId={nodeId} selected={selected}>
        {bodyComponentKey === 'call_saved_workflows' ? (
          <CallSavedWorkflowsNode nodeId={nodeId} isOpen={isOpen} />
        ) : (
          <InvocationNode nodeId={nodeId} isOpen={isOpen} />
        )}
      </NodeWrapper>
    </InvocationNodeContextProvider>
  );
};

export default memo(InvocationNodeWrapper);

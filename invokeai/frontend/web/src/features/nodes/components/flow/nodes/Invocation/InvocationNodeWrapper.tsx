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

import InvocationNodeUnknownFallback from './InvocationNodeUnknownFallback';

const InvocationNodeWrapper = (props: NodeProps<Node<InvocationNodeData>>) => {
  const { data, selected } = props;
  const { id: nodeId, type, isOpen, label } = data;
  const templates = useStore($templates);
  const hasTemplate = useMemo(() => Boolean(templates[type]), [templates, type]);
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
      <NodeWrapper nodeId={nodeId} selected={selected}>
        <InvocationNodeUnknownFallback nodeId={nodeId} isOpen={isOpen} label={label} type={type} />
      </NodeWrapper>
    );
  }

  return (
    <NodeWrapper nodeId={nodeId} selected={selected}>
      <InvocationNode nodeId={nodeId} isOpen={isOpen} />
    </NodeWrapper>
  );
};

export default memo(InvocationNodeWrapper);

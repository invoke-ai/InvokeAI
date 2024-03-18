import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import InvocationNode from 'features/nodes/components/flow/nodes/Invocation/InvocationNode';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import type { InvocationNodeData } from 'features/nodes/types/invocation';
import { memo, useMemo } from 'react';
import type { NodeProps } from 'reactflow';

import InvocationNodeUnknownFallback from './InvocationNodeUnknownFallback';

const InvocationNodeWrapper = (props: NodeProps<InvocationNodeData>) => {
  const { data, selected } = props;
  const { id: nodeId, type, isOpen, label } = data;

  const hasTemplateSelector = useMemo(
    () => createSelector(selectNodesSlice, (nodes) => Boolean(nodes.templates[type])),
    [type]
  );

  const hasTemplate = useAppSelector(hasTemplateSelector);

  if (!hasTemplate) {
    return (
      <InvocationNodeUnknownFallback nodeId={nodeId} isOpen={isOpen} label={label} type={type} selected={selected} />
    );
  }

  return <InvocationNode nodeId={nodeId} isOpen={isOpen} label={label} type={type} selected={selected} />;
};

export default memo(InvocationNodeWrapper);

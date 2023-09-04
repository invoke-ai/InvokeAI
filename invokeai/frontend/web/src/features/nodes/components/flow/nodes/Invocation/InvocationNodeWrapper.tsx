import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { InvocationNodeData } from 'features/nodes/types/types';
import { memo, useMemo } from 'react';
import { NodeProps } from 'reactflow';
import InvocationNode from '../Invocation/InvocationNode';
import InvocationNodeUnknownFallback from './InvocationNodeUnknownFallback';

const InvocationNodeWrapper = (props: NodeProps<InvocationNodeData>) => {
  const { data, selected } = props;
  const { id: nodeId, type, isOpen, label } = data;

  const hasTemplateSelector = useMemo(
    () =>
      createSelector(stateSelector, ({ nodes }) =>
        Boolean(nodes.nodeTemplates[type])
      ),
    [type]
  );

  const nodeTemplate = useAppSelector(hasTemplateSelector);

  if (!nodeTemplate) {
    return (
      <InvocationNodeUnknownFallback
        nodeId={nodeId}
        isOpen={isOpen}
        label={label}
        type={type}
        selected={selected}
      />
    );
  }

  return (
    <InvocationNode
      nodeId={nodeId}
      isOpen={isOpen}
      label={label}
      type={type}
      selected={selected}
    />
  );
};

export default memo(InvocationNodeWrapper);

import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import InvocationNode from 'features/nodes/components/flow/nodes/Invocation/InvocationNode';
import { InvocationNodeData } from 'features/nodes/types/invocation';
import { memo, useMemo } from 'react';
import { NodeProps } from 'reactflow';
import InvocationNodeUnknownFallback from './InvocationNodeUnknownFallback';

const InvocationNodeWrapper = (props: NodeProps<InvocationNodeData>) => {
  const { data, selected } = props;
  const { id: nodeId, type, isOpen, label } = data;

  const hasTemplateSelector = useMemo(
    () =>
      createMemoizedSelector(stateSelector, ({ nodes }) =>
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

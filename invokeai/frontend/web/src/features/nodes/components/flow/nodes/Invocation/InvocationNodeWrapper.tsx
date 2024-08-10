import { useStore } from '@nanostores/react';
import type { NodeProps } from '@xyflow/react';
import { useAppSelector } from 'app/store/storeHooks';
import { InvocationNodeComponent } from 'features/nodes/components/flow/nodes/Invocation/InvocationNodeComponent';
import { $templates } from 'features/nodes/store/nodesSlice';
import type { InvocationNode } from 'features/nodes/types/invocation';
import { memo, useMemo } from 'react';

import InvocationNodeUnknownFallback from './InvocationNodeUnknownFallback';

const InvocationNodeWrapper = (props: NodeProps<InvocationNode>) => {
  const { data, selected } = props;
  const { id: nodeId, type, isOpen, label } = data;
  const templates = useStore($templates);
  const hasTemplate = useMemo(() => Boolean(templates[type]), [templates, type]);
  const nodeExists = useAppSelector((s) => Boolean(s.nodes.present.nodes.find((n) => n.id === nodeId)));

  if (!nodeExists) {
    return null;
  }

  if (!hasTemplate) {
    return (
      <InvocationNodeUnknownFallback nodeId={nodeId} isOpen={isOpen} label={label} type={type} selected={selected} />
    );
  }

  return <InvocationNodeComponent nodeId={nodeId} isOpen={isOpen} label={label} type={type} selected={selected} />;
};

export default memo(InvocationNodeWrapper);

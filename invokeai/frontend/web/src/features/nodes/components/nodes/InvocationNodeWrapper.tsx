import { useAppSelector } from 'app/store/storeHooks';
import { makeTemplateSelector } from 'features/nodes/store/util/makeTemplateSelector';
import { InvocationNodeData } from 'features/nodes/types/types';
import { memo, useMemo } from 'react';
import { NodeProps } from 'reactflow';
import InvocationNode from '../Invocation/InvocationNode';
import UnknownNodeFallback from '../Invocation/UnknownNodeFallback';

const InvocationNodeWrapper = (props: NodeProps<InvocationNodeData>) => {
  const { data } = props;
  const { type } = data;

  const templateSelector = useMemo(() => makeTemplateSelector(type), [type]);

  const nodeTemplate = useAppSelector(templateSelector);

  if (!nodeTemplate) {
    return <UnknownNodeFallback nodeProps={props} />;
  }

  return <InvocationNode nodeProps={props} nodeTemplate={nodeTemplate} />;
};

export default memo(InvocationNodeWrapper);

import { NODE_MIN_WIDTH } from 'features/nodes/types/constants';
import { memo } from 'react';
import { NodeResizeControl, NodeResizerProps } from 'reactflow';

// this causes https://github.com/invoke-ai/InvokeAI/issues/4140
// not using it for now

const NodeResizer = (props: NodeResizerProps) => {
  const { ...rest } = props;
  return (
    <NodeResizeControl
      style={{
        position: 'absolute',
        border: 'none',
        background: 'transparent',
        width: 15,
        height: 15,
        bottom: 0,
        right: 0,
      }}
      minWidth={NODE_MIN_WIDTH}
      {...rest}
    ></NodeResizeControl>
  );
};

export default memo(NodeResizer);

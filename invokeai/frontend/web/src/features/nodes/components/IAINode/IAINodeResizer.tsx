import { NodeResizeControl, NodeResizerProps } from 'reactflow';

export default function IAINodeResizer(props: NodeResizerProps) {
  const { ...rest } = props;
  return (
    <NodeResizeControl
      style={{
        position: 'relative',
        border: 'none',
        background: 'none',
        width: 10,
        height: 10,
        top: 10,
      }}
      minWidth={350}
      {...rest}
    ></NodeResizeControl>
  );
}

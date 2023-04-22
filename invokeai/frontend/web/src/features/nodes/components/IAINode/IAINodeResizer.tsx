import { NodeResizeControl, NodeResizerProps } from 'reactflow';

export default function IAINodeResizer(props: NodeResizerProps) {
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
      minWidth={350}
      {...rest}
    ></NodeResizeControl>
  );
}

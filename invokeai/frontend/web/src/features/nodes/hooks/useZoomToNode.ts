import { useReactFlow } from '@xyflow/react';
import { useCallback } from 'react';

export const useZoomToNode = () => {
  const flow = useReactFlow();
  const zoomToNode = useCallback(
    (nodeId: string) => {
      flow.fitView({ duration: 300, maxZoom: 1.5, nodes: [{ id: nodeId }] });
    },
    [flow]
  );
  return zoomToNode;
};

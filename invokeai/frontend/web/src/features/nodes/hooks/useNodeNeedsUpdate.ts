import { useNodeData } from 'features/nodes/hooks/useNodeData';
import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { getNeedsUpdate } from 'features/nodes/util/node/nodeUpdate';
import { useMemo } from 'react';

export const useNodeNeedsUpdate = (nodeId: string) => {
  const data = useNodeData(nodeId);
  const template = useNodeTemplate(nodeId);
  const needsUpdate = useMemo(() => getNeedsUpdate(data, template), [data, template]);
  return needsUpdate;
};

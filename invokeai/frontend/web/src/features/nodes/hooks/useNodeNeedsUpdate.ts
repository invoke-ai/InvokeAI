import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { useNodeType } from 'features/nodes/hooks/useNodeType';
import { useNodeVersion } from 'features/nodes/hooks/useNodeVersion';
import { useMemo } from 'react';

export const useNodeNeedsUpdate = (nodeId: string) => {
  const type = useNodeType(nodeId);
  const version = useNodeVersion(nodeId);
  const template = useNodeTemplate(nodeId);
  const needsUpdate = useMemo(() => {
    if (type !== template.type) {
      return true;
    }
    return version !== template.version;
  }, [template.type, template.version, type, version]);
  return needsUpdate;
};

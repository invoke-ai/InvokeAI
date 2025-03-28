import { useNodeType } from 'features/nodes/hooks/useNodeType';
import { useNodeVersion } from 'features/nodes/hooks/useNodeVersion';
import { useMemo } from 'react';

import { useNodeTemplateOrThrow } from './useNodeTemplateOrThrow';

export const useNodeNeedsUpdate = (nodeId: string) => {
  const type = useNodeType(nodeId);
  const version = useNodeVersion(nodeId);
  const template = useNodeTemplateOrThrow(nodeId);
  const needsUpdate = useMemo(() => {
    if (type !== template.type) {
      return true;
    }
    return version !== template.version;
  }, [template.type, template.version, type, version]);
  return needsUpdate;
};

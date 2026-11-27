import { useNodeType } from 'features/nodes/hooks/useNodeType';
import { useNodeVersion } from 'features/nodes/hooks/useNodeVersion';
import { useMemo } from 'react';

import { useNodeTemplateOrThrow } from './useNodeTemplateOrThrow';

export const useNodeNeedsUpdate = () => {
  const type = useNodeType();
  const version = useNodeVersion();
  const template = useNodeTemplateOrThrow();
  const needsUpdate = useMemo(() => {
    if (type !== template.type) {
      return true;
    }
    return version !== template.version;
  }, [template.type, template.version, type, version]);
  return needsUpdate;
};

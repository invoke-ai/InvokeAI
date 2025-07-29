import { useIsExecutableNode } from 'features/nodes/hooks/useIsBatchNode';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useMemo } from 'react';

import { useNodeHasImageOutput } from './useNodeHasImageOutput';

export const useWithFooter = () => {
  const hasImageOutput = useNodeHasImageOutput();
  const isExecutableNode = useIsExecutableNode();
  const isCacheEnabled = useFeatureStatus('invocationCache');
  const withFooter = useMemo(
    () => isExecutableNode && (hasImageOutput || isCacheEnabled),
    [hasImageOutput, isCacheEnabled, isExecutableNode]
  );
  return withFooter;
};

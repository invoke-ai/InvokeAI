import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useMemo } from 'react';

import { useNodeHasImageOutput } from './useNodeHasImageOutput';

export const useWithFooter = (nodeId: string) => {
  const hasImageOutput = useNodeHasImageOutput(nodeId);
  const isCacheEnabled = useFeatureStatus('invocationCache');
  const withFooter = useMemo(() => hasImageOutput || isCacheEnabled, [hasImageOutput, isCacheEnabled]);
  return withFooter;
};

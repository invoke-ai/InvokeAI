import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useMemo } from 'react';

import { useHasImageOutput } from './useHasImageOutput';

export const useWithFooter = (nodeId: string) => {
  const hasImageOutput = useHasImageOutput(nodeId);
  const isCacheEnabled = useFeatureStatus('invocationCache');
  const withFooter = useMemo(() => hasImageOutput || isCacheEnabled, [hasImageOutput, isCacheEnabled]);
  return withFooter;
};

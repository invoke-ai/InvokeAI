import { useAppSelector } from 'app/store/storeHooks';
import { buildSelectIsStagingBySessionId } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useMemo } from 'react';

import { useCanvasSessionId } from './useCanvasSessionId';

export const useCanvasIsStaging = () => {
  const sessionId = useCanvasSessionId();
  const selectIsStagingBySessionIdSelector = useMemo(() => buildSelectIsStagingBySessionId(sessionId), [sessionId]);

  return useAppSelector(selectIsStagingBySessionIdSelector);
};

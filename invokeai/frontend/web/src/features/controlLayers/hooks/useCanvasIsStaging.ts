import { useAppSelector } from 'app/store/storeHooks';
import {
  buildSelectIsStagingBySessionId,
  selectActiveCanvasStagingAreaSessionId,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useMemo } from 'react';

export const useCanvasIsStaging = () => {
  const sessionId = useAppSelector(selectActiveCanvasStagingAreaSessionId);
  const selectIsStagingBySessionIdSelector = useMemo(() => buildSelectIsStagingBySessionId(sessionId), [sessionId]);

  return useAppSelector(selectIsStagingBySessionIdSelector);
};

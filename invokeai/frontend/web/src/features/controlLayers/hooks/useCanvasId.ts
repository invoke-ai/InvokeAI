import { useAppSelector } from 'app/store/storeHooks';
import { useScopedCanvasIdSafe } from 'features/controlLayers/contexts/CanvasInstanceContextProvider';
import { selectActiveCanvasId } from 'features/controlLayers/store/selectors';

export const useCanvasId = () => {
  const scopedCanvasId = useScopedCanvasIdSafe();
  const canvasId = useAppSelector(selectActiveCanvasId);

  return scopedCanvasId ?? canvasId;
};

import { useAppSelector } from 'app/store/storeHooks';
import { selectActiveCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';

export const useActiveCanvasIsStaging = () => {
  return useAppSelector(selectActiveCanvasIsStaging);
};

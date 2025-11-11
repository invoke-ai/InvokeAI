import { useActiveCanvasIsStaging } from 'features/controlLayers/hooks/useCanvasIsStaging';

export const useIsBboxSizeLocked = () => {
  const isStaging = useActiveCanvasIsStaging();
  return isStaging;
};

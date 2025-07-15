import { useSDXLModels } from 'services/api/hooks/modelsByType';

export const useIsRefinerAvailable = () => {
  const [sdxlModels] = useSDXLModels();

  return Boolean(sdxlModels.length);
};

import { useRefinerModels } from 'services/api/hooks/modelsByType';

export const useIsRefinerAvailable = () => {
  const [refinerModels] = useRefinerModels();

  return Boolean(refinerModels.length);
};

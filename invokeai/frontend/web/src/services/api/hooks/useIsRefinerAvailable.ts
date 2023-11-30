import { REFINER_BASE_MODELS } from 'services/api/constants';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';

export const useIsRefinerAvailable = () => {
  const { isRefinerAvailable } = useGetMainModelsQuery(REFINER_BASE_MODELS, {
    selectFromResult: ({ data }) => ({
      isRefinerAvailable: data ? data.ids.length > 0 : false,
    }),
  });

  return isRefinerAvailable;
};

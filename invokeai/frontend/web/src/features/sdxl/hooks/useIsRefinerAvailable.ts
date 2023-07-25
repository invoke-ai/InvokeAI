import { useGetSDXLRefinerModelsQuery } from 'services/api/endpoints/models';

export const useIsRefinerAvailable = () => {
  const { isRefinerAvailable } = useGetSDXLRefinerModelsQuery(undefined, {
    selectFromResult: ({ data }) => ({
      isRefinerAvailable: data ? data.ids.length > 0 : false,
    }),
  });

  return isRefinerAvailable;
};
